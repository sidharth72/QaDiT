"""
PyTorch/XLA training loop for Kaggle TPU v5e-8 (8 chips, data parallel).

Run it directly on a Kaggle TPU VM session (from a notebook cell, as a SCRIPT):

    !python train_xla.py --data /kaggle/input/.../audiocaps-precomputed-cache \
                         --out  /kaggle/working/runs/dit_b2

Do NOT call torch_xla.launch from a notebook cell, and do NOT touch
xm.xla_device() in the notebook before launching. Kaggle sets
TPU_PROCESS_ADDRESSES which this file clears automatically.

What is different from the GPU/CPU loop in train.py, and WHY
------------------------------------------------------------
XLA compiles the training step into a fixed graph and replays it.  Anything
that (a) changes the graph between steps or (b) reads device tensors back to
the host mid-step destroys performance.  Concretely, this port:

  1. MULTI-PROCESS DATA PARALLELISM - `xmp.spawn` forks one process
     per TPU chip (8 on v5e-8).  Each process owns one device; gradients are
     all-reduced across chips before the optimizer step. Gradient accumulation
     keeps the effective global batch large without exceeding v5e's 16 GB HBM.
  2. `MpDeviceLoader` - wraps the CPU DataLoader, prefetches batches to the
     device in background threads, and inserts the `mark_step()` graph cut
     after every iteration (we never call it manually).
  3. NO `.item()` / `float()` / `.any()` IN THE HOT PATH - every host read
     forces the lazy graph to execute early and stalls the pipeline.  Losses
     are logged through `xm.add_step_closure`, which materialises values only
     when the step's execution finishes anyway - and only every log_every
     steps on the master process.
  4. GRADIENT CLIPPING lives BETWEEN `xm.reduce_gradients` (the cross-chip
     all-reduce) and `optimizer.step()`, so we clip the true global gradient,
     identically on every replica.
  5. bf16 AUTOCAST - matmuls/attention run in bfloat16 (v5e's native format),
     while master weights, optimizer state and the EMA stay fp32.
  6. FIXED SHAPES EVERYWHERE - the precomputed tensors already have constant
     shapes and the DataLoader uses drop_last=True, so XLA compiles the step
     graph once and reuses it for the whole run.
  7. IDENTICAL INIT + PER-RANK DATA - the model is built from one shared seed
     (all replicas start with byte-identical weights, so the all-reduce keeps
     them in lockstep), while the data sampler and the device RNG (noise,
     timesteps, CFG dropout) are seeded per rank.
  8. VALIDATION is sharded without padding or duplicate examples, reduced
     across all chips, and uses a fixed RNG stream without perturbing training.
  9. CHECKPOINTS via `xm.save` - gathers tensors to CPU and writes atomically
     from the master process; data/RNG/W&B state resumes cleanly.

Everything about the MODEL and the MATH (DiT, diffusion, REPA, EMA, LR
schedule, CFG dropout) is identical to train.py - only the execution
harness changes.
"""

import argparse
import math
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import wandb
import shutil
import os

# Kaggle injects TPU_PROCESS_ADDRESSES="local" (1 address). Multi-process XLA
# needs one address per chip (8 on v5e-8). If left set, init fails with:
#   Expected 8 worker addresses, got 1
# Only pop THIS variable (see Kaggle product-feedback/473974). Do not strip
# other TPU_* vars — that can break rank discovery (int(None) on LOCAL_RANK).
os.environ.pop("TPU_PROCESS_ADDRESSES", None)

# Notebook / accelerate leftovers make torch_xla.launch() think a distributed
# job is already running, then it does int(os.environ["LOCAL_RANK"]) → None.
for _dist_env in (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "TORCHELASTIC_RUN_ID",
):
    os.environ.pop(_dist_env, None)

os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

from config import Config
from dataset import (DistributedEvalSampler, PrecomputedAudioCaps,
                     ShardDistributedSampler)
from diffusion import Diffusion
from dit import RepaProjector, repa_loss
from train import EMA, build_model, lr_lambda_factory, repa_lambda


def collate_tensors_only(batch: list[dict]) -> dict:
    """Like dataset.collate but WITHOUT the caption strings.

    MpDeviceLoader tries to ship every leaf of the batch to the TPU; strings
    can't go and training never needs them, so they are dropped here.
    """
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "repa": torch.stack([b["repa"] for b in batch]),
    }


def upload_ckpt_folder(
    local_dir: str | Path,
    repo_id: str,
    token: str,
    path_in_repo: str | None = None,
    commit_message: str = "upload checkpoint",
):
    """
    Upload a local folder (or a single checkpoint directory) to a HF model repo.
    repo_id example: "your-username/audio-dit-checkpoints"
    path_in_repo: subfolder inside the repo, e.g. "runs/dit_b2"
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        # Ignore temp / incomplete files if any
        ignore_patterns=["*.tmp", "*.lock"],
    )
    print(f"[hf] uploaded {local_dir} -> {repo_id}"
          + (f"/{path_in_repo}" if path_in_repo else ""))


def init_wandb(cfg, is_master: bool, project: str = "audio-dit",
               run_name: str | None = None, enabled: bool = True,
               resume_id: str | None = None):
    """Create the W&B run on the master ordinal only."""
    if not (enabled and is_master):
        return None
    run = wandb.init(
        project=project,
        name=run_name or "dit_b2_tpu_v5e8",
        config=cfg.as_dict(),
        resume="allow" if resume_id else None,
        id=resume_id,
    )
    print(f"[wandb] run: {run.url}", flush=True)
    return run


def make_log_fn(is_master, start_step, t0, sched, global_batch: int,
                use_wandb: bool = True):
    """XLA-safe logger used via xm.add_step_closure (print + wandb)."""
    def log_fn(step, loss, diff, repa, lam):
        # Materialise tensors HERE — the graph has already executed.
        loss_v = float(loss)
        diff_v = float(diff)
        repa_v = float(repa)
        lam_v = float(lam)
        lr = sched.get_last_lr()[0]
        rate = (step - start_step + 1) / max(time.time() - t0, 1e-6)
        sample_rate = rate * global_batch

        if not is_master:
            return

        print(
            f"step {step:>7d} | loss {loss_v:.4f} "
            f"(diff {diff_v:.4f}, repa {repa_v:.4f}, lam {lam_v:.3f}) "
            f"| lr {lr:.2e} | {rate:.2f} updates/s "
            f"| {sample_rate:.1f} samples/s global",
            flush=True,
        )
        if use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/loss_total": loss_v,
                    "train/loss_diff": diff_v,
                    "train/loss_repa": repa_v,
                    "train/repa_lambda": lam_v,
                    "train/lr": lr,
                    "perf/updates_per_s": rate,
                    "perf/samples_per_s_global": sample_rate,
                },
                step=step,
            )
    return log_fn


def _per_sample_repa_loss(projected: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """Negative token-mean cosine similarity for each sample."""
    projected = F.normalize(projected.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(projected * target).sum(dim=-1).mean(dim=-1)


@torch.no_grad()
def validate(model, projector, ema, diffusion, device_loader, device,
             rank: int, step: int, lam: float, seed: int) -> dict[str, float]:
    """Run deterministic, sample-weighted validation across every XLA rank."""
    # Materialize the preceding optimizer update before snapshotting its RNG.
    torch_xla.sync()
    train_rng_state = xm.get_rng_state(device=device)
    xm.set_rng_state(seed + rank, device=device)

    model.eval()
    projector.eval()
    ema.model.eval()
    stats = torch.zeros(5, dtype=torch.float32, device=device)
    try:
        for batch in device_loader:
            z0 = batch["latent"]
            text_emb = batch["text_emb"]
            text_mask = batch["text_mask"]
            y_star = batch["repa"]
            batch_size = z0.shape[0]

            t = diffusion.sample_timesteps(batch_size, device)
            eps = torch.randn_like(z0)
            z_t = diffusion.add_noise(z0, t, eps)
            v_target = diffusion.v_target(z0, t, eps)

            with torch.autocast("xla", dtype=torch.bfloat16):
                v_pred, hidden = model(
                    z_t, t.float(), text_emb, text_mask,
                    drop_mask=None, return_repa_hidden=True)
                diff_per_sample = (
                    v_pred.float() - v_target
                ).square().flatten(1).mean(dim=1)
                repa_per_sample = _per_sample_repa_loss(
                    projector(hidden), y_star)
                ema_pred = ema.model(
                    z_t, t.float(), text_emb, text_mask, drop_mask=None)
                ema_diff_per_sample = (
                    ema_pred.float() - v_target
                ).square().flatten(1).mean(dim=1)

            total_per_sample = diff_per_sample + lam * repa_per_sample
            stats[0] += total_per_sample.sum()
            stats[1] += diff_per_sample.sum()
            stats[2] += repa_per_sample.sum()
            stats[3] += ema_diff_per_sample.sum()
            stats[4] += batch_size

        stats = xm.all_reduce(xm.REDUCE_SUM, stats)
        torch_xla.sync()
        total, diff, repa, ema_diff, count = stats.cpu().tolist()
        if count <= 0:
            raise RuntimeError("validation split contains no examples")
        return {
            "loss_total": total / count,
            "loss_diff": diff / count,
            "loss_repa": repa / count,
            "ema_loss_diff": ema_diff / count,
            "num_samples": count,
        }
    finally:
        xm.set_rng_state(train_rng_state, device=device)
        model.train()
        projector.train()


def log_validation(metrics: dict[str, float], step: int, is_master: bool,
                   use_wandb: bool):
    if not is_master:
        return
    print(
        f"[val] step {step:>7d} | loss {metrics['loss_total']:.4f} "
        f"(diff {metrics['loss_diff']:.4f}, repa {metrics['loss_repa']:.4f}) "
        f"| ema diff {metrics['ema_loss_diff']:.4f} "
        f"| n={int(metrics['num_samples'])}",
        flush=True,
    )
    if use_wandb and wandb.run is not None:
        wandb.log(
            {
                "val/loss_total": metrics["loss_total"],
                "val/loss_diff": metrics["loss_diff"],
                "val/loss_repa": metrics["loss_repa"],
                "val/ema_loss_diff": metrics["ema_loss_diff"],
            },
            step=step,
        )

def save_ckpt(
    out_dir: Path,
    model,
    projector,
    ema,
    opt,
    sched,
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
    """
    Atomically save complete resume state and optionally upload it.
    """
    out_dir = Path(out_dir)
    ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
    temp_path = ckpt_path.with_suffix(".pt.tmp")
    is_master = xm.is_master_ordinal()

    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    xm.rendezvous(f"ckpt_dir_{step}")

    # Ensure pending random ops have advanced the running XLA RNG before it is
    # captured. Every rank contributes its state to the master checkpoint.
    torch_xla.sync()
    local_rng_state = xm.get_rng_state()
    rng_states = xm.mesh_reduce(
        f"ckpt_rng_{step}", local_rng_state, lambda values: list(values))

    payload = {
        "checkpoint_version": 2,
        "model": model.state_dict(),
        "projector": projector.state_dict(),
        "ema": ema.model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "config": cfg.as_dict(),
        "data_state": {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
        },
        "runtime": runtime_state,
        "xla_rng_states": rng_states,
        "best_val": best_val,
        "wandb_run_id": (
            wandb.run.id if is_master and wandb.run is not None else None),
    }
    xm.save(payload, str(temp_path))
    xm.rendezvous(f"ckpt_temp_{step}")

    if is_master:
        os.replace(temp_path, ckpt_path)
    xm.rendezvous(f"ckpt_committed_{step}")

    if is_master:
        token_ok = bool(hf_token)  # treat "" as missing
        should_upload = (
            bool(hf_repo_id)
            and token_ok
            and upload_every > 0
            and (force_upload or (step // max(cfg.train.ckpt_every, 1)) % upload_every == 0)
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
                # Other ranks must still be allowed to reach the rendezvous.
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

    xm.rendezvous(f"ckpt_done_{step}")
    return ckpt_path


def validate_resume_config(saved: dict | None, current: dict):
    """Reject silent architecture/training changes when resuming."""
    if saved is None:
        return
    if saved != current:
        raise ValueError(
            "checkpoint config differs from the current Config(); restore the "
            "original config before resuming or start a new run")


def _mp_fn(index: int, args: argparse.Namespace):
    cfg = Config()
    device = xm.xla_device()
    rank = xr.global_ordinal()
    world = xr.world_size()
    is_master = xm.is_master_ordinal()

    effective_global_batch = (
        args.batch_size * args.grad_accum_steps * world)
    runtime_state = {
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "world_size": world,
        "effective_global_batch": effective_global_batch,
    }
    if is_master:
        print(
            f"[xla] world_size={world}  microbatch/chip={args.batch_size}  "
            f"grad_accum={args.grad_accum_steps}  "
            f"effective global batch={effective_global_batch}",
            flush=True,
        )

    # ---------------- data (sharded across the 8 chips) -------------------- #
    ds = PrecomputedAudioCaps(args.data, "train")
    sampler = ShardDistributedSampler(
        ds, num_replicas=world, rank=rank,
        samples_per_step=args.batch_size * args.grad_accum_steps,
        seed=cfg.train.seed,
    )
    full_batches_per_epoch = sampler.num_samples // args.batch_size
    if full_batches_per_epoch == 0:
        raise ValueError(
            "training split is too small for one effective global batch "
            f"({effective_global_batch} samples)")
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_tensors_only,
                        persistent_workers=args.workers > 0)

    val_ds = PrecomputedAudioCaps(
        args.data, args.val_split, latent_scale=ds.latent_scale)
    val_sampler = DistributedEvalSampler(
        val_ds, num_replicas=world, rank=rank)
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, sampler=val_sampler,
        num_workers=args.workers, drop_last=False,
        collate_fn=collate_tensors_only,
        persistent_workers=args.workers > 0,
    )
    if is_master:
        print(
            f"[xla] train={len(ds)} samples, "
            f"{full_batches_per_epoch} microbatches/epoch/chip; "
            f"validation={len(val_ds)} samples; "
            f"train latent scale={ds.latent_scale:.4f}",
            flush=True,
        )

    # ---------------- model / optimizer ------------------------------------ #
    # Same seed on every rank BEFORE building the model.
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg).to(device)
    projector = RepaProjector(cfg.dit.hidden_size,
                              cfg.pretrained.repa_dim).to(device)
    diffusion = Diffusion(cfg.diffusion.num_train_steps, cfg.diffusion.schedule,
                          cfg.diffusion.logit_normal_mean,
                          cfg.diffusion.logit_normal_std).to(device)

    if is_master:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[xla] DiT parameters: {n_params:.1f}M")

    params = list(model.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(cfg.train.warmup_steps, cfg.train.total_steps))
    ema = EMA(model, cfg.train.ema_decay)

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
            for key in ("batch_size", "grad_accum_steps", "world_size"):
                if saved_runtime.get(key) != runtime_state[key]:
                    raise ValueError(
                        f"resume runtime mismatch for {key}: "
                        f"{saved_runtime.get(key)} != {runtime_state[key]}")
        model.load_state_dict(resume_ckpt["model"])
        projector.load_state_dict(resume_ckpt["projector"])
        ema.model.load_state_dict(resume_ckpt["ema"])
        opt.load_state_dict(resume_ckpt["opt"])
        sched.load_state_dict(resume_ckpt["sched"])
        start_step = resume_ckpt["step"] + 1
        data_state = resume_ckpt.get("data_state")
        if data_state is None:
            consumed_microbatches = start_step * args.grad_accum_steps
            epoch, batch_in_epoch = divmod(
                consumed_microbatches, full_batches_per_epoch)
            if is_master:
                print(
                    "[xla] legacy checkpoint: inferred data cursor; exact old "
                    "sampler order cannot be reconstructed",
                    flush=True,
                )
        else:
            epoch = int(data_state["epoch"])
            batch_in_epoch = int(data_state["batch_in_epoch"])
        best_val = resume_ckpt.get("best_val", best_val)
        if is_master:
            print(f"[xla] resumed from {args.resume} at step {start_step}")

    # Different RNG per rank for noise / timesteps / CFG dropout.
    torch.manual_seed(cfg.train.seed * 1000 + rank)
    saved_rng_states = (
        resume_ckpt.get("xla_rng_states") if resume_ckpt is not None else None)
    if saved_rng_states is not None and rank < len(saved_rng_states):
        xm.set_rng_state(int(saved_rng_states[rank]), device=device)
    else:
        xm.set_rng_state(cfg.train.seed * 1000 + rank, device=device)

    out_dir = Path(args.out)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- W&B + logging ---------------------------------------- #
    wandb_resume_id = (
        resume_ckpt.get("wandb_run_id") if resume_ckpt is not None else None)
    wandb_run = init_wandb(
        cfg, is_master, project=args.wandb_project,
        run_name=args.wandb_run_name, enabled=args.wandb_enabled,
        resume_id=wandb_resume_id,
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "xla_runtime": runtime_state,
                "validation": {
                    "split": args.val_split,
                    "batch_size": args.val_batch_size,
                    "every": args.val_every,
                    "seed": args.val_seed,
                },
            },
            allow_val_change=True,
        )
    t0 = time.time()
    log_fn = make_log_fn(is_master, start_step, t0, sched,
                         effective_global_batch,
                         use_wandb=args.wandb_enabled)

    # ---------------- the loop --------------------------------------------- #
    model.train()
    projector.train()
    step = start_step
    last_saved_step = resume_ckpt["step"] if resume_ckpt is not None else None
    done = step >= cfg.train.total_steps
    opt.zero_grad(set_to_none=True)

    while not done:
        sampler.set_epoch(epoch)
        sampler.set_start_index(batch_in_epoch * args.batch_size)
        device_loader = pl.MpDeviceLoader(loader, device)
        micro_in_update = 0
        loss_sum = diff_sum = repa_sum = None

        for batch in device_loader:

            z0 = batch["latent"]
            text_emb = batch["text_emb"]
            text_mask = batch["text_mask"]
            y_star = batch["repa"]
            B = z0.shape[0]

            drop_mask = torch.rand(B, device=device) < cfg.diffusion.p_uncond
            t = diffusion.sample_timesteps(B, device)
            eps = torch.randn_like(z0)
            z_t = diffusion.add_noise(z0, t, eps)
            v_target = diffusion.v_target(z0, t, eps)

            lam = repa_lambda(step, cfg.train.repa_weight,
                              cfg.train.repa_decay_steps)

            with torch.autocast("xla", dtype=torch.bfloat16):
                v_pred, h_l = model(z_t, t.float(), text_emb, text_mask,
                                    drop_mask=drop_mask,
                                    return_repa_hidden=True)
                loss_diff = F.mse_loss(v_pred.float(), v_target)
                loss_repa = repa_loss(projector(h_l).float(), y_star)
                loss = loss_diff + lam * loss_repa

            (loss / args.grad_accum_steps).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
            diff_sum = (loss_diff.detach() if diff_sum is None
                        else diff_sum + loss_diff.detach())
            repa_sum = (loss_repa.detach() if repa_sum is None
                        else repa_sum + loss_repa.detach())
            micro_in_update += 1
            batch_in_epoch += 1

            if micro_in_update < args.grad_accum_steps:
                continue

            xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            opt.zero_grad(set_to_none=True)

            if step % cfg.train.log_every == 0:
                train_metrics = torch.stack((
                    loss_sum / args.grad_accum_steps,
                    diff_sum / args.grad_accum_steps,
                    repa_sum / args.grad_accum_steps,
                ))
                train_metrics = xm.all_reduce(
                    xm.REDUCE_SUM, train_metrics, scale=1.0 / world)
                xm.add_step_closure(
                    log_fn,
                    args=(step, train_metrics[0], train_metrics[1],
                          train_metrics[2], lam),
                )

            next_epoch = epoch
            next_batch = batch_in_epoch
            if next_batch == full_batches_per_epoch:
                next_epoch += 1
                next_batch = 0

            if (args.val_every > 0 and step > 0
                    and step % args.val_every == 0):
                val_metrics = validate(
                    model, projector, ema, diffusion,
                    pl.MpDeviceLoader(val_loader, device),
                    device, rank, step, lam, args.val_seed,
                )
                log_validation(
                    val_metrics, step, is_master, args.wandb_enabled)
                if val_metrics["ema_loss_diff"] < best_val["ema_loss_diff"]:
                    best_val = {
                        "ema_loss_diff": val_metrics["ema_loss_diff"],
                        "step": step,
                    }

            # The final checkpoint is handled below with force_upload=True.
            if (step > 0 and step % cfg.train.ckpt_every == 0
                    and step + 1 < cfg.train.total_steps):
                save_ckpt(
                    out_dir,
                    model, projector, ema, opt, sched, step, cfg,
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
                    print(f"[xla] checkpoint at step {step}", flush=True)
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
                    "epoch ended with a partial gradient accumulation window")
            epoch += 1
            batch_in_epoch = 0

    # Final save + force HF upload so the last weights are durable.
    final_step = step - 1
    if final_step >= 0 and final_step != last_saved_step:
        save_ckpt(
            out_dir,
            model, projector, ema, opt, sched, final_step, cfg,
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
        if wandb.run is not None:
            wandb.finish()
        print(
            f"[xla] done at step {final_step}; checkpoints: {out_dir}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="precompute cache root containing train/ and validation/")
    ap.add_argument("--out", type=str, default="/kaggle/working/runs/dit_b2")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers per process")
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size",
                    type=int, default=4,
                    help="per-chip TPU microbatch (default: 4 for v5e HBM)")
    ap.add_argument(
        "--grad-accum-steps", "--grad_accum_steps",
        dest="grad_accum_steps", type=int, default=8,
        help="microbatches per optimizer update (4 x 8 x 8 = 256 global)",
    )
    ap.add_argument("--val-split", "--val_split", dest="val_split",
                    type=str, default="validation")
    ap.add_argument("--val-batch-size", "--val_batch_size",
                    dest="val_batch_size", type=int, default=4)
    ap.add_argument("--val-every", "--val_every", dest="val_every",
                    type=int, default=2000,
                    help="optimizer-update interval; <=0 disables validation")
    ap.add_argument("--val-seed", "--val_seed", dest="val_seed",
                    type=int, default=17_029)

    ap.add_argument("--hf_repo_id", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--upload_every", type=int, default=5)
    ap.add_argument("--keep_local", type=int, default=1)
    ap.add_argument("--path_in_repo", type=str, default="checkpoints")
    ap.add_argument("--wandb_project", type=str, default="audio-dit")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_enabled", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Use --wandb-enabled / --no-wandb-enabled")
    args = ap.parse_args()
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        ap.error("--batch-size and --grad-accum-steps must be positive")
    if args.val_batch_size <= 0:
        ap.error("--val-batch-size must be positive")
    if args.workers < 0:
        ap.error("--workers cannot be negative")
    for split in ("train", args.val_split):
        meta_path = Path(args.data) / split / "meta.json"
        if not meta_path.exists():
            ap.error(f"missing precomputed split metadata: {meta_path}")

    # Re-clear in case the parent notebook exported these into the shell env.
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    for key in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "TORCHELASTIC_RUN_ID",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    # Use xmp.spawn directly on Kaggle. torch_xla.launch() can take a
    # "already distributed" branch when WORLD_SIZE/LOCAL_RANK are present and
    # then crash with: int() argument ... not 'NoneType'.
    xmp.spawn(_mp_fn, args=(args,), nprocs=None)


if __name__ == "__main__":
    main()
