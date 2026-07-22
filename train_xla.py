"""
PyTorch/XLA training loop for Kaggle TPU v5e-8 (8 chips, data parallel).

Run it directly on a Kaggle TPU VM session:

    python train_xla.py --data /kaggle/input/audiocaps-precomputed \
                        --out  /kaggle/working/runs/dit_b2

What is different from the GPU/CPU loop in train.py, and WHY
------------------------------------------------------------
XLA compiles the training step into a fixed graph and replays it.  Anything
that (a) changes the graph between steps or (b) reads device tensors back to
the host mid-step destroys performance.  Concretely, this port:

  1. MULTI-PROCESS DATA PARALLELISM - `torch_xla.launch` forks one process
     per TPU chip (8 on v5e-8).  Each process owns one device; gradients are
     all-reduced across chips before the optimizer step.  cfg.train.batch_size
     is PER CHIP, so the global batch is 8x that (32 -> 256 global).
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
  8. CHECKPOINTS via `xm.save` - gathers tensors to CPU and writes from the
     master process only; Kaggle sessions are preemptible, so we save often
     and `--resume` cleanly.

Everything about the MODEL and the MATH (DiT, diffusion, REPA, EMA, LR
schedule, CFG dropout) is identical to train.py - only the execution
harness changes.
"""

import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import wandb
import shutil
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

from config import Config
from dataset import PrecomputedAudioCaps
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


def make_log_fn(is_master, start_step, t0, sched, use_wandb: bool = True):
    """XLA-safe logger used via xm.add_step_closure (print + wandb)."""
    def log_fn(step, loss, diff, repa, lam):
        # Materialise tensors HERE — the graph has already executed.
        loss_v = float(loss)
        diff_v = float(diff)
        repa_v = float(repa)
        lam_v = float(lam)
        lr = sched.get_last_lr()[0]
        rate = (step - start_step + 1) / max(time.time() - t0, 1e-6)

        if not is_master:
            return

        print(
            f"step {step:>7d} | loss {loss_v:.4f} "
            f"(diff {diff_v:.4f}, repa {repa_v:.4f}, lam {lam_v:.3f}) "
            f"| lr {lr:.2e} | {rate:.2f} it/s/chip",
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
                    "perf/it_per_s_per_chip": rate,
                },
                step=step,
            )
    return log_fn

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
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    upload_every: int = 5,
    keep_local: int = 1,
    path_in_repo: str = "checkpoints",
    force_upload: bool = False,
):
    """
    Save checkpoint, prune old local files, optionally upload to Hugging Face.

    - Writes: out_dir / f"ckpt_{step:07d}.pt"
    - Deletes older ckpt_*.pt so /kaggle/working stays small
    - Uploads every `upload_every` local saves (or always if force_upload=True)
    """
    out_dir = Path(out_dir)
    ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
    is_master = xm.is_master_ordinal()

    payload = {
        "model": model.state_dict(),
        "projector": projector.state_dict(),
        "ema": ema.model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "config": cfg.as_dict(),
    }
    xm.save(payload, str(ckpt_path))

    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpts = sorted(out_dir.glob("ckpt_*.pt"))
        if keep_local >= 0 and len(ckpts) > keep_local:
            for old in ckpts[:-keep_local]:
                try:
                    old.unlink()
                    print(f"[ckpt] deleted old local: {old.name}", flush=True)
                except OSError as e:
                    print(f"[ckpt] failed to delete {old}: {e}", flush=True)

        token_ok = bool(hf_token)  # treat "" as missing
        should_upload = (
            bool(hf_repo_id)
            and token_ok
            and upload_every > 0
            and (force_upload or (step // max(cfg.train.ckpt_every, 1)) % upload_every == 0)
        )
        if should_upload:
            stage = out_dir / "_hf_upload"
            stage.mkdir(exist_ok=True)
            for f in stage.glob("*"):
                if f.is_file():
                    f.unlink()
            staged = stage / ckpt_path.name
            shutil.copy2(ckpt_path, staged)
            upload_ckpt_folder(
                local_dir=stage,
                repo_id=hf_repo_id,
                token=hf_token,
                path_in_repo=path_in_repo,
                commit_message=f"ckpt step {step}",
            )

    xm.rendezvous("ckpt_saved")


def _mp_fn(index: int, args: argparse.Namespace):
    cfg = Config()
    device = xm.xla_device()
    rank = xr.global_ordinal()
    world = xr.world_size()
    is_master = xm.is_master_ordinal()

    if is_master:
        print(f"[xla] world_size={world}  per-chip batch={cfg.train.batch_size}  "
              f"global batch={cfg.train.batch_size * world}")

    # ---------------- data (sharded across the 8 chips) -------------------- #
    ds = PrecomputedAudioCaps(args.data, "train")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, sampler=sampler,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_tensors_only,
                        persistent_workers=args.workers > 0)
    device_loader = pl.MpDeviceLoader(loader, device)
    if is_master:
        print(f"[xla] {len(ds)} samples, {len(loader)} steps/epoch/chip, "
              f"latent scale = {ds.latent_scale:.4f}")

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
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        projector.load_state_dict(ckpt["projector"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        if is_master:
            print(f"[xla] resumed from {args.resume} at step {start_step}")

    # Different RNG per rank for noise / timesteps / CFG dropout.
    torch.manual_seed(cfg.train.seed * 1000 + rank)
    xm.set_rng_state(cfg.train.seed * 1000 + rank, device=device)

    out_dir = Path(args.out)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- W&B + logging ---------------------------------------- #
    init_wandb(cfg, is_master, project=args.wandb_project,
               run_name=args.wandb_run_name, enabled=args.wandb_enabled)
    t0 = time.time()
    log_fn = make_log_fn(is_master, start_step, t0, sched,
                         use_wandb=args.wandb_enabled)

    # ---------------- the loop --------------------------------------------- #
    model.train()
    step = start_step
    epoch = 0
    done = False

    while not done:
        sampler.set_epoch(epoch)
        for batch in device_loader:
            if step >= cfg.train.total_steps:
                done = True
                break

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

            opt.zero_grad(set_to_none=True)
            loss.backward()
            xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)

            if step % cfg.train.log_every == 0:
                xm.add_step_closure(
                    log_fn, args=(step, loss.detach(), loss_diff.detach(),
                                  loss_repa.detach(), lam))

            if step > 0 and step % cfg.train.ckpt_every == 0:
                save_ckpt(
                    out_dir,
                    model, projector, ema, opt, sched, step, cfg,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=args.hf_token,
                    upload_every=args.upload_every,
                    keep_local=args.keep_local,
                    path_in_repo=args.path_in_repo,
                )
                if is_master:
                    print(f"[xla] checkpoint at step {step}", flush=True)

            step += 1
        epoch += 1

    # Final save + force HF upload so the last weights are durable.
    save_ckpt(
        out_dir,
        model, projector, ema, opt, sched, step - 1, cfg,
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
        print(f"[xla] done - final checkpoint in {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="precompute cache root (folder containing train/)")
    ap.add_argument("--out", type=str, default="/kaggle/working/runs/dit_b2")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers per process")

    ap.add_argument("--hf_repo_id", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--upload_every", type=int, default=5)
    ap.add_argument("--keep_local", type=int, default=1)
    ap.add_argument("--path_in_repo", type=str, default="checkpoints")
    ap.add_argument("--wandb_project", type=str, default="audio-dit")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_enabled", type=bool, default=True)
    args = ap.parse_args()

    # Fork one process per TPU chip (8 on v5e-8).  `torch_xla.launch` is the
    # modern entry point; xmp.spawn is the fallback for older torch_xla.
    try:
        torch_xla.launch(_mp_fn, args=(args,))
    except AttributeError:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(args,))


if __name__ == "__main__":
    main()
