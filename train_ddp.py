"""
Two-GPU PyTorch DDP trainer for the text-to-audio latent DiT.

Designed for Kaggle's 2 x NVIDIA T4 accelerator. Launch it as a script:

    torchrun --standalone --nproc_per_node=2 train_ddp.py \
        --data /kaggle/input/audiocaps-precomputed \
        --out /kaggle/working/runs/dit_b2

Each torchrun process owns one GPU. DDP averages gradients across the two
replicas, FP16 autocast and GradScaler reduce memory use, and gradient
accumulation preserves a large effective global batch.
"""

import argparse
from contextlib import ExitStack
from datetime import timedelta
from importlib import import_module
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    wandb = import_module("wandb")
except ImportError:
    wandb = None

from config import Config
from dataset import (
    DistributedEvalSampler,
    PrecomputedAudioCaps,
    ShardDistributedSampler,
)
from diffusion import Diffusion
from dit import RepaProjector, repa_loss
from train import EMA, build_model, lr_lambda_factory, repa_lambda


def collate_tensors_only(batch: list[dict]) -> dict:
    """Stack training tensors and omit captions, which the trainer never uses."""
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "repa": torch.stack([b["repa"] for b in batch]),
    }


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def upload_ckpt_folder(
    local_dir: str | Path,
    repo_id: str,
    token: str,
    path_in_repo: str | None = None,
    commit_message: str = "upload checkpoint",
):
    """Upload a checkpoint folder to a Hugging Face model repository."""
    try:
        hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when --hf-repo-id is set"
        ) from exc

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)
    api = hub.HfApi(token=token)
    hub.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=["*.tmp", "*.lock"],
    )
    print(
        f"[hf] uploaded {local_dir} -> {repo_id}"
        + (f"/{path_in_repo}" if path_in_repo else ""),
        flush=True,
    )


def init_wandb(
    cfg,
    is_master: bool,
    project: str = "audio-dit",
    run_name: str | None = None,
    enabled: bool = True,
    resume_id: str | None = None,
    api_key: str | None = None,
):
    """Create one W&B run on global rank zero.

    Authenticates with the API key immediately so wandb never prompts for a
    browser/API-key choice in notebooks or headless jobs.
    """
    if not (enabled and is_master):
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed; install requirements.txt or pass "
            "--no-wandb-enabled"
        )
    key = api_key or os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError(
            "W&B is enabled but no API key was found. Pass --wandb-api-key, "
            "set WANDB_API_KEY, or use --no-wandb-enabled"
        )
    os.environ["WANDB_API_KEY"] = key
    # Keep the session non-interactive even if wandb tries to re-auth.
    os.environ.setdefault("WANDB_SILENT", "true")
    wandb.login(key=key, relogin=True, anonymous="never")
    run = wandb.init(
        project=project,
        name=run_name or "dit_b2_2xt4",
        config=cfg.as_dict(),
        resume="allow" if resume_id else None,
        id=resume_id,
        settings=wandb.Settings(quiet=True),
    )
    print(f"[wandb] run: {run.url}", flush=True)
    return run


def log_train(
    step: int,
    metrics: torch.Tensor,
    lam: float,
    sched,
    start_step: int,
    t0: float,
    global_batch: int,
    use_wandb: bool,
):
    """Print and optionally report globally averaged training metrics."""
    loss_v, diff_v, repa_v = metrics.cpu().tolist()
    lr = sched.get_last_lr()[0]
    rate = (step - start_step + 1) / max(time.time() - t0, 1e-6)
    sample_rate = rate * global_batch
    print(
        f"step {step:>7d} | loss {loss_v:.4f} "
        f"(diff {diff_v:.4f}, repa {repa_v:.4f}, lam {lam:.3f}) "
        f"| lr {lr:.2e} | {rate:.2f} updates/s "
        f"| {sample_rate:.1f} samples/s global",
        flush=True,
    )
    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.log(
            {
                "train/loss_total": loss_v,
                "train/loss_diff": diff_v,
                "train/loss_repa": repa_v,
                "train/repa_lambda": lam,
                "train/lr": lr,
                "perf/updates_per_s": rate,
                "perf/samples_per_s_global": sample_rate,
            },
            step=step,
        )


def _per_sample_repa_loss(
    projected: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Negative token-mean cosine similarity for each sample."""
    projected = F.normalize(projected.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(projected * target).sum(dim=-1).mean(dim=-1)


@torch.no_grad()
def validate(
    model,
    projector,
    ema,
    diffusion,
    loader,
    device,
    rank: int,
    lam: float,
    seed: int,
    amp_enabled: bool,
) -> dict[str, float]:
    """Run deterministic, sample-weighted validation across all GPU ranks."""
    raw_model = unwrap(model)
    raw_projector = unwrap(projector)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)

    raw_model.eval()
    raw_projector.eval()
    ema.model.eval()
    stats = torch.zeros(5, dtype=torch.float64, device=device)
    try:
        for batch in loader:
            z0 = batch["latent"].to(device, non_blocking=True)
            text_emb = batch["text_emb"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            y_star = batch["repa"].to(device, non_blocking=True)
            batch_size = z0.shape[0]

            t = diffusion.sample_timesteps(batch_size, device)
            eps = torch.randn_like(z0)
            z_t = diffusion.add_noise(z0, t, eps)
            v_target = diffusion.v_target(z0, t, eps)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                v_pred, hidden = raw_model(
                    z_t,
                    t.float(),
                    text_emb,
                    text_mask,
                    drop_mask=None,
                    return_repa_hidden=True,
                )
                diff_per_sample = (
                    v_pred.float() - v_target
                ).square().flatten(1).mean(dim=1)
                repa_per_sample = _per_sample_repa_loss(
                    raw_projector(hidden), y_star
                )
                ema_pred = ema.model(
                    z_t, t.float(), text_emb, text_mask, drop_mask=None
                )
                ema_diff_per_sample = (
                    ema_pred.float() - v_target
                ).square().flatten(1).mean(dim=1)

            total_per_sample = diff_per_sample + lam * repa_per_sample
            stats[0] += total_per_sample.double().sum()
            stats[1] += diff_per_sample.double().sum()
            stats[2] += repa_per_sample.double().sum()
            stats[3] += ema_diff_per_sample.double().sum()
            stats[4] += batch_size

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, diff_loss, repa, ema_diff, count = stats.cpu().tolist()
        if count <= 0:
            raise RuntimeError("validation split contains no examples")
        return {
            "loss_total": total / count,
            "loss_diff": diff_loss / count,
            "loss_repa": repa / count,
            "ema_loss_diff": ema_diff / count,
            "num_samples": count,
        }
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)
        raw_model.train()
        raw_projector.train()


def log_validation(
    metrics: dict[str, float],
    step: int,
    is_master: bool,
    use_wandb: bool,
):
    if not is_master:
        return
    print(
        f"[val] step {step:>7d} | loss {metrics['loss_total']:.4f} "
        f"(diff {metrics['loss_diff']:.4f}, repa {metrics['loss_repa']:.4f}) "
        f"| ema diff {metrics['ema_loss_diff']:.4f} "
        f"| n={int(metrics['num_samples'])}",
        flush=True,
    )
    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.log(
            {
                "val/loss_total": metrics["loss_total"],
                "val/loss_diff": metrics["loss_diff"],
                "val/loss_repa": metrics["loss_repa"],
                "val/ema_loss_diff": metrics["ema_loss_diff"],
            },
            step=step,
        )


def validate_resume_config(saved: dict | None, current: dict):
    """Reject silent architecture or training changes when resuming."""
    if saved is not None and saved != current:
        raise ValueError(
            "checkpoint config differs from the current Config(); restore the "
            "original config before resuming or start a new run"
        )


def gather_rng_states(device) -> list[dict]:
    """Collect CPU and CUDA RNG state from every rank for exact resume."""
    local_state = {
        "cpu": torch.get_rng_state().cpu(),
        "cuda": torch.cuda.get_rng_state(device).cpu(),
    }
    states: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return states


def save_ckpt(
    out_dir: Path,
    model,
    projector,
    ema,
    opt,
    sched,
    scaler,
    step: int,
    cfg,
    *,
    epoch: int,
    batch_in_epoch: int,
    runtime_state: dict,
    best_val: dict,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    upload_every: int = 5,
    keep_local: int = 1,
    path_in_repo: str = "checkpoints",
    force_upload: bool = False,
):
    """Atomically save complete DDP resume state from global rank zero."""
    rank = dist.get_rank()
    is_master = rank == 0
    out_dir = Path(out_dir)
    ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
    temp_path = ckpt_path.with_suffix(".pt.tmp")

    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    rng_states = gather_rng_states(torch.cuda.current_device())

    if is_master:
        payload = {
            "checkpoint_version": 3,
            "model": unwrap(model).state_dict(),
            "projector": unwrap(projector).state_dict(),
            "ema": ema.model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": cfg.as_dict(),
            "data_state": {
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
            },
            "runtime": runtime_state,
            "rng_states": rng_states,
            "best_val": best_val,
            "wandb_run_id": (
                wandb.run.id
                if wandb is not None and wandb.run is not None
                else None
            ),
        }
        torch.save(payload, temp_path)
        os.replace(temp_path, ckpt_path)
    dist.barrier()

    if is_master:
        should_upload = (
            bool(hf_repo_id)
            and bool(hf_token)
            and upload_every > 0
            and (
                force_upload
                or (step // max(cfg.train.ckpt_every, 1)) % upload_every == 0
            )
        )
        upload_succeeded = False
        if should_upload:
            stage = out_dir / "_hf_upload"
            try:
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir()
                shutil.copy2(ckpt_path, stage / ckpt_path.name)
                upload_ckpt_folder(
                    local_dir=stage,
                    repo_id=hf_repo_id,
                    token=hf_token,
                    path_in_repo=path_in_repo,
                    commit_message=f"ckpt step {step}",
                )
                upload_succeeded = True
            except Exception as exc:
                print(f"[hf] checkpoint upload failed: {exc}", flush=True)
            finally:
                shutil.rmtree(stage, ignore_errors=True)

        ckpts = sorted(out_dir.glob("ckpt_*.pt"))
        if keep_local >= 0:
            effective_keep = keep_local
            if keep_local == 0 and not upload_succeeded:
                effective_keep = 1
                print(
                    "[ckpt] preserving newest local checkpoint because no "
                    "durable upload completed",
                    flush=True,
                )
            excess = max(0, len(ckpts) - effective_keep)
            for old in ckpts[:excess]:
                try:
                    old.unlink()
                    print(f"[ckpt] deleted old local: {old.name}", flush=True)
                except OSError as exc:
                    print(f"[ckpt] failed to delete {old}: {exc}", flush=True)
    dist.barrier()
    return ckpt_path


def train(args: argparse.Namespace):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=2),
    )

    rank = dist.get_rank()
    world = dist.get_world_size()
    is_master = rank == 0
    device = torch.device("cuda", local_rank)

    if world != 2:
        raise RuntimeError(
            f"this trainer expects exactly 2 processes/GPUs, but got {world}; "
            "launch with torchrun --standalone --nproc_per_node=2"
        )

    cfg = Config()
    effective_global_batch = args.batch_size * args.grad_accum_steps * world
    runtime_state = {
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "world_size": world,
        "effective_global_batch": effective_global_batch,
        "amp": args.amp,
    }
    if is_master:
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"[ddp] world_size={world}  GPU={gpu_name}  "
            f"microbatch/GPU={args.batch_size}  "
            f"grad_accum={args.grad_accum_steps}  "
            f"effective global batch={effective_global_batch}  "
            f"precision={'fp16' if args.amp else 'fp32'}",
            flush=True,
        )

    # The train and validation folders are independent precomputed splits.
    ds = PrecomputedAudioCaps(args.data, "train")
    sampler = ShardDistributedSampler(
        ds,
        num_replicas=world,
        rank=rank,
        samples_per_step=args.batch_size * args.grad_accum_steps,
        seed=cfg.train.seed,
    )
    full_batches_per_epoch = sampler.num_samples // args.batch_size
    if full_batches_per_epoch == 0:
        raise ValueError(
            "training split is too small for one effective global batch "
            f"({effective_global_batch} samples)"
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_tensors_only,
        persistent_workers=args.workers > 0,
    )

    val_ds = PrecomputedAudioCaps(
        args.data, args.val_split, latent_scale=ds.latent_scale
    )
    if len(val_ds) == 0:
        raise ValueError(
            f"validation split '{args.val_split}' is empty under {args.data}"
        )
    val_sampler = DistributedEvalSampler(
        val_ds, num_replicas=world, rank=rank
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_tensors_only,
        persistent_workers=args.workers > 0,
    )
    runtime_state["data_signature"] = {
        "train_meta": ds.meta,
        "validation_meta": val_ds.meta,
        "sampler_num_samples_per_rank": sampler.num_samples,
        "full_batches_per_epoch_per_rank": full_batches_per_epoch,
    }
    if is_master:
        print(
            f"[ddp] train={len(ds)} samples, "
            f"{full_batches_per_epoch} microbatches/epoch/GPU; "
            f"validation={len(val_ds)} samples; "
            f"train latent scale={ds.latent_scale:.4f}",
            flush=True,
        )

    # Identical seed before construction gives every rank identical weights.
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed(cfg.train.seed)
    raw_model = build_model(cfg).to(device)
    raw_projector = RepaProjector(
        cfg.dit.hidden_size, cfg.pretrained.repa_dim
    ).to(device)
    diffusion = Diffusion(
        cfg.diffusion.num_train_steps,
        cfg.diffusion.schedule,
        cfg.diffusion.logit_normal_mean,
        cfg.diffusion.logit_normal_std,
    ).to(device)

    if is_master:
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        print(f"[ddp] DiT parameters: {n_params:.1f}M", flush=True)

    params = list(raw_model.parameters()) + list(raw_projector.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(cfg.train.warmup_steps, cfg.train.total_steps)
    )
    ema = EMA(raw_model, cfg.train.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    epoch = 0
    batch_in_epoch = 0
    best_val = {"ema_loss_diff": math.inf, "step": None}
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        validate_resume_config(resume_ckpt.get("config"), cfg.as_dict())
        saved_runtime = resume_ckpt.get("runtime")
        if saved_runtime is not None:
            for key in (
                "batch_size",
                "grad_accum_steps",
                "world_size",
                "amp",
            ):
                if saved_runtime.get(key) != runtime_state[key]:
                    raise ValueError(
                        f"resume runtime mismatch for {key}: "
                        f"{saved_runtime.get(key)} != {runtime_state[key]}"
                    )
            saved_data_signature = saved_runtime.get("data_signature")
            if saved_data_signature is None:
                if is_master:
                    print(
                        "[ddp] checkpoint has no dataset signature; exact "
                        "cache identity could not be validated",
                        flush=True,
                    )
            elif saved_data_signature != runtime_state["data_signature"]:
                raise ValueError(
                    "resume dataset mismatch: train/validation metadata or "
                    "sampler geometry differs from the checkpoint"
                )
        raw_model.load_state_dict(resume_ckpt["model"])
        raw_projector.load_state_dict(resume_ckpt["projector"])
        ema.model.load_state_dict(resume_ckpt["ema"])
        opt.load_state_dict(resume_ckpt["opt"])
        sched.load_state_dict(resume_ckpt["sched"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_step = resume_ckpt["step"] + 1
        data_state = resume_ckpt.get("data_state")
        if data_state is None:
            consumed_microbatches = start_step * args.grad_accum_steps
            epoch, batch_in_epoch = divmod(
                consumed_microbatches, full_batches_per_epoch
            )
            if is_master:
                print(
                    "[ddp] legacy checkpoint: inferred data cursor; exact old "
                    "sampler order cannot be reconstructed",
                    flush=True,
                )
        else:
            epoch = int(data_state["epoch"])
            batch_in_epoch = int(data_state["batch_in_epoch"])
        best_val = resume_ckpt.get("best_val", best_val)
        if is_master:
            print(f"[ddp] resumed from {args.resume} at step {start_step}")

    # DDP wraps only after checkpoint loading to keep state-dict keys portable.
    model = DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    projector = DDP(
        raw_projector,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    # Training randomness differs per rank; checkpointed states restore it.
    torch.manual_seed(cfg.train.seed * 1000 + rank)
    torch.cuda.manual_seed(cfg.train.seed * 1000 + rank)
    saved_rng_states = (
        resume_ckpt.get("rng_states") if resume_ckpt is not None else None
    )
    if saved_rng_states is not None and rank < len(saved_rng_states):
        torch.set_rng_state(saved_rng_states[rank]["cpu"])
        torch.cuda.set_rng_state(saved_rng_states[rank]["cuda"], device)

    out_dir = Path(args.out)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    wandb_resume_id = (
        resume_ckpt.get("wandb_run_id") if resume_ckpt is not None else None
    )
    wandb_run = init_wandb(
        cfg,
        is_master,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        enabled=args.wandb_enabled,
        resume_id=wandb_resume_id,
        api_key=args.wandb_api_key,
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "ddp_runtime": runtime_state,
                "validation": {
                    "split": args.val_split,
                    "batch_size": args.val_batch_size,
                    "every": args.val_every,
                    "seed": args.val_seed,
                },
            },
            allow_val_change=True,
        )

    model.train()
    projector.train()
    step = start_step
    last_saved_step = resume_ckpt["step"] if resume_ckpt is not None else None
    done = step >= cfg.train.total_steps
    opt.zero_grad(set_to_none=True)
    t0 = time.time()

    while not done:
        sampler.set_epoch(epoch)
        sampler.set_start_index(batch_in_epoch * args.batch_size)
        micro_in_update = 0
        loss_sum = diff_sum = repa_sum = None

        for batch in loader:
            z0 = batch["latent"].to(device, non_blocking=True)
            text_emb = batch["text_emb"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            y_star = batch["repa"].to(device, non_blocking=True)
            batch_size = z0.shape[0]

            drop_mask = (
                torch.rand(batch_size, device=device) < cfg.diffusion.p_uncond
            )
            t = diffusion.sample_timesteps(batch_size, device)
            eps = torch.randn_like(z0)
            z_t = diffusion.add_noise(z0, t, eps)
            v_target = diffusion.v_target(z0, t, eps)
            lam = repa_lambda(
                step, cfg.train.repa_weight, cfg.train.repa_decay_steps
            )

            is_update_microbatch = (
                micro_in_update + 1 == args.grad_accum_steps
            )
            with ExitStack() as stack:
                if not is_update_microbatch:
                    stack.enter_context(model.no_sync())
                    stack.enter_context(projector.no_sync())
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=args.amp,
                ):
                    v_pred, hidden = model(
                        z_t,
                        t.float(),
                        text_emb,
                        text_mask,
                        drop_mask=drop_mask,
                        return_repa_hidden=True,
                    )
                    loss_diff = F.mse_loss(v_pred.float(), v_target)
                    loss_repa = repa_loss(projector(hidden).float(), y_star)
                    loss = loss_diff + lam * loss_repa
                scaler.scale(loss / args.grad_accum_steps).backward()

            loss_sum = (
                loss.detach() if loss_sum is None else loss_sum + loss.detach()
            )
            diff_sum = (
                loss_diff.detach()
                if diff_sum is None
                else diff_sum + loss_diff.detach()
            )
            repa_sum = (
                loss_repa.detach()
                if repa_sum is None
                else repa_sum + loss_repa.detach()
            )
            micro_in_update += 1
            batch_in_epoch += 1

            if not is_update_microbatch:
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            optimizer_ran = scaler.get_scale() >= old_scale
            opt.zero_grad(set_to_none=True)

            if not optimizer_ran:
                if is_master:
                    print(
                        f"[amp] non-finite gradients at step {step}; "
                        "skipped update and reduced loss scale",
                        flush=True,
                    )
                micro_in_update = 0
                loss_sum = diff_sum = repa_sum = None
                continue

            sched.step()
            ema.update(unwrap(model))

            if step % cfg.train.log_every == 0:
                train_metrics = torch.stack(
                    (
                        loss_sum / args.grad_accum_steps,
                        diff_sum / args.grad_accum_steps,
                        repa_sum / args.grad_accum_steps,
                    )
                )
                dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
                train_metrics /= world
                if is_master:
                    log_train(
                        step,
                        train_metrics,
                        lam,
                        sched,
                        start_step,
                        t0,
                        effective_global_batch,
                        args.wandb_enabled,
                    )

            next_epoch = epoch
            next_batch = batch_in_epoch
            if next_batch == full_batches_per_epoch:
                next_epoch += 1
                next_batch = 0

            if (
                args.val_every > 0
                and step > 0
                and step % args.val_every == 0
            ):
                val_metrics = validate(
                    model,
                    projector,
                    ema,
                    diffusion,
                    val_loader,
                    device,
                    rank,
                    lam,
                    args.val_seed,
                    args.amp,
                )
                log_validation(
                    val_metrics, step, is_master, args.wandb_enabled
                )
                if val_metrics["ema_loss_diff"] < best_val["ema_loss_diff"]:
                    best_val = {
                        "ema_loss_diff": val_metrics["ema_loss_diff"],
                        "step": step,
                    }

            if (
                step > 0
                and step % cfg.train.ckpt_every == 0
                and step + 1 < cfg.train.total_steps
            ):
                save_ckpt(
                    out_dir,
                    model,
                    projector,
                    ema,
                    opt,
                    sched,
                    scaler,
                    step,
                    cfg,
                    epoch=next_epoch,
                    batch_in_epoch=next_batch,
                    runtime_state=runtime_state,
                    best_val=best_val,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=args.hf_token,
                    upload_every=args.upload_every,
                    keep_local=args.keep_local,
                    path_in_repo=args.path_in_repo,
                )
                if is_master:
                    print(f"[ddp] checkpoint at step {step}", flush=True)
                last_saved_step = step

            step += 1
            micro_in_update = 0
            loss_sum = diff_sum = repa_sum = None
            if step >= cfg.train.total_steps:
                done = True
                epoch, batch_in_epoch = next_epoch, next_batch
                break

        if not done:
            if micro_in_update != 0:
                raise RuntimeError(
                    "epoch ended with a partial gradient accumulation window"
                )
            epoch += 1
            batch_in_epoch = 0

    final_step = step - 1
    if final_step >= 0 and final_step != last_saved_step:
        save_ckpt(
            out_dir,
            model,
            projector,
            ema,
            opt,
            sched,
            scaler,
            final_step,
            cfg,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            runtime_state=runtime_state,
            best_val=best_val,
            hf_repo_id=args.hf_repo_id,
            hf_token=args.hf_token,
            upload_every=args.upload_every,
            keep_local=args.keep_local,
            path_in_repo=args.path_in_repo,
            force_upload=True,
        )
    if is_master:
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        print(
            f"[ddp] done at step {final_step}; checkpoints: {out_dir}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="precompute cache root containing train/ and validation/",
    )
    ap.add_argument(
        "--out", type=str, default="/kaggle/working/runs/dit_b2"
    )
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader workers per GPU process",
    )
    ap.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=2,
        help="microbatch per T4 GPU",
    )
    ap.add_argument(
        "--grad-accum-steps",
        "--grad_accum_steps",
        dest="grad_accum_steps",
        type=int,
        default=64,
        help="microbatches per update (2 x 64 x 2 = 256 global by default)",
    )
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use FP16 autocast and GradScaler (default: enabled)",
    )
    ap.add_argument(
        "--val-split",
        "--val_split",
        dest="val_split",
        type=str,
        default="validation",
    )
    ap.add_argument(
        "--val-batch-size",
        "--val_batch_size",
        dest="val_batch_size",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--val-every",
        "--val_every",
        dest="val_every",
        type=int,
        default=2000,
        help="optimizer-update interval; <=0 disables validation",
    )
    ap.add_argument(
        "--val-seed",
        "--val_seed",
        dest="val_seed",
        type=int,
        default=17_029,
    )
    ap.add_argument("--hf-repo-id", "--hf_repo_id", dest="hf_repo_id")
    ap.add_argument(
        "--hf-token",
        "--hf_token",
        dest="hf_token",
        default=os.environ.get("HF_TOKEN"),
    )
    ap.add_argument(
        "--upload-every",
        "--upload_every",
        dest="upload_every",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--keep-local",
        "--keep_local",
        dest="keep_local",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--path-in-repo",
        "--path_in_repo",
        dest="path_in_repo",
        default="checkpoints",
    )
    ap.add_argument(
        "--wandb-project",
        "--wandb_project",
        dest="wandb_project",
        default="audio-dit",
    )
    ap.add_argument(
        "--wandb-run-name",
        "--wandb_run_name",
        dest="wandb_run_name",
        default=None,
    )
    ap.add_argument(
        "--wandb-api-key",
        "--wandb_api_key",
        dest="wandb_api_key",
        default=os.environ.get("WANDB_API_KEY"),
        help="W&B API key; defaults to WANDB_API_KEY env (no interactive login)",
    )
    ap.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use --wandb-enabled / --no-wandb-enabled",
    )
    args = ap.parse_args()

    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        ap.error("--batch-size and --grad-accum-steps must be positive")
    if args.val_batch_size <= 0:
        ap.error("--val-batch-size must be positive")
    if args.workers < 0:
        ap.error("--workers cannot be negative")
    if "LOCAL_RANK" not in os.environ:
        ap.error(
            "launch this script with torchrun --standalone "
            "--nproc_per_node=2 train_ddp.py ..."
        )
    if not torch.cuda.is_available():
        ap.error("CUDA is required for the DDP trainer")
    for split in ("train", args.val_split):
        meta_path = Path(args.data) / split / "meta.json"
        if not meta_path.exists():
            ap.error(f"missing precomputed split metadata: {meta_path}")
    return args


def main():
    args = parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
