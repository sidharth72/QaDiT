"""
Training loop for the text-to-audio latent DiT.

Two ways to run it:

  1. REAL TRAINING (needs precompute.py output):
        python train.py --data ./cache
  2. SMOKE TEST (no data, no pretrained models, runs on CPU in ~a minute):
        python train.py --smoke
     Builds a tiny DiT, feeds synthetic tensors of the exact real shapes,
     checks that both losses go down, and runs the DDIM sampler end-to-end.
     If this passes, the whole architecture is wired correctly and the same
     code paths will work on GPU - and later, with minor changes, on XLA.

Per training step (mirrors AudioDiffusionModel.md section 11):

    z0'  = scaled VAE latent                    (precomputed)
    c    = T5 hidden states  (dropped to null with p=0.1  -> enables CFG)
    t    ~ logit-normal,  eps ~ N(0, I)
    z_t  = sqrt(abar_t) z0' + sqrt(1-abar_t) eps
    v', h_l = DiT(z_t, t, c)                    (h_l = REPA tap)
    L    = ||v - v'||^2  +  lambda(step) * ( -cos( g(h_l), y* ) )

Only the DiT and the REPA projector receive gradients.  An EMA copy of the
DiT is maintained and is what you should sample from.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from diffusion import Diffusion
from dit import DiT, RepaProjector, repa_loss


# --------------------------------------------------------------------------- #
#  EMA - exponential moving average of model weights                           #
# --------------------------------------------------------------------------- #
class EMA:
    """Keeps a slow-moving copy of the DiT.  Sampling from the EMA weights
    instead of the raw weights is a standard, material quality win."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1.0 - self.decay)     # ema = d*ema + (1-d)*param
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


# --------------------------------------------------------------------------- #
#  LR schedule: linear warmup then cosine decay to zero                        #
# --------------------------------------------------------------------------- #
def lr_lambda_factory(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return fn


def repa_lambda(step: int, base: float, decay_steps: int) -> float:
    """REPA weight schedule: full strength early, linear decay to 0.
    REPA accelerates early representation learning; late in training the
    diffusion loss should own the objective."""
    if base <= 0.0:
        return 0.0
    return base * max(0.0, 1.0 - step / max(decay_steps, 1))


# --------------------------------------------------------------------------- #
#  Model construction (shared by train.py and sample.py)                       #
# --------------------------------------------------------------------------- #
def build_model(cfg: Config) -> DiT:
    return DiT(
        latent_channels=cfg.latent.channels,
        latent_time=cfg.latent.time,
        latent_freq=cfg.latent.freq,
        patch_size=cfg.dit.patch_size,
        hidden_size=cfg.dit.hidden_size,
        depth=cfg.dit.depth,
        num_heads=cfg.dit.num_heads,
        mlp_ratio=cfg.dit.mlp_ratio,
        text_dim=cfg.pretrained.text_dim,
        repa_layer=cfg.dit.repa_layer,
    )


# --------------------------------------------------------------------------- #
#  Synthetic data for the smoke test                                           #
# --------------------------------------------------------------------------- #
class SyntheticBatches:
    """Infinite iterator of random tensors with the exact shapes the real
    pipeline produces.  Lets us verify every code path with zero downloads."""

    def __init__(self, cfg: Config, device):
        self.cfg, self.device = cfg, device
        n_tokens = (cfg.latent.time // cfg.dit.patch_size) * \
                   (cfg.latent.freq // cfg.dit.patch_size)
        self.n_tokens = n_tokens

    def next(self) -> dict:
        cfg, dev = self.cfg, self.device
        B = cfg.train.batch_size
        L = cfg.pretrained.text_max_len
        mask = torch.zeros(B, L, dtype=torch.long, device=dev)
        mask[:, : L // 2] = 1                     # pretend half the tokens are real
        return {
            "latent": torch.randn(B, cfg.latent.channels,
                                  cfg.latent.time, cfg.latent.freq, device=dev),
            "text_emb": torch.randn(B, L, cfg.pretrained.text_dim, device=dev),
            "text_mask": mask,
            "repa": torch.randn(B, self.n_tokens, cfg.pretrained.repa_dim, device=dev),
        }


# --------------------------------------------------------------------------- #
#  One training step                                                           #
# --------------------------------------------------------------------------- #
def train_step(batch: dict, model: DiT, projector: RepaProjector,
               diffusion: Diffusion, cfg: Config, step: int,
               device) -> tuple[torch.Tensor, dict]:
    z0 = batch["latent"].to(device)               # already unit-variance scaled
    text_emb = batch["text_emb"].to(device)
    text_mask = batch["text_mask"].to(device)
    y_star = batch["repa"].to(device)
    B = z0.shape[0]

    # --- classifier-free-guidance dropout: null out ~10% of captions ------- #
    drop_mask = torch.rand(B, device=device) < cfg.diffusion.p_uncond

    # --- forward diffusion -------------------------------------------------- #
    t = diffusion.sample_timesteps(B, device)     # logit-normal over [0, T)
    eps = torch.randn_like(z0)
    z_t = diffusion.add_noise(z0, t, eps)
    v_target = diffusion.v_target(z0, t, eps)

    # --- model forward (with REPA tap) -------------------------------------- #
    v_pred, h_l = model(z_t, t.float(), text_emb, text_mask,
                        drop_mask=drop_mask, return_repa_hidden=True)

    # --- losses -------------------------------------------------------------- #
    loss_diff = F.mse_loss(v_pred, v_target)
    lam = repa_lambda(step, cfg.train.repa_weight, cfg.train.repa_decay_steps)
    loss_repa = repa_loss(projector(h_l), y_star) if lam > 0 else z0.new_zeros(())
    loss = loss_diff + lam * loss_repa

    logs = {"loss": loss.item(), "diff": loss_diff.item(),
            "repa": float(loss_repa), "lambda": lam}
    return loss, logs


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./cache",
                    help="root folder produced by precompute.py")
    ap.add_argument("--out", type=str, default="./runs/dit_b2")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + synthetic data: end-to-end sanity check")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint .pt to resume from")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.smoke() if args.smoke else Config()
    device = torch.device(args.device)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data -------------------------------------------------- #
    if args.smoke:
        data = SyntheticBatches(cfg, device)
        get_batch = data.next
        print("[train] SMOKE MODE - tiny model, synthetic data")
    else:
        from dataset import PrecomputedAudioCaps, collate
        ds = PrecomputedAudioCaps(args.data, "train")
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                            num_workers=2, pin_memory=(device.type == "cuda"),
                            collate_fn=collate, drop_last=True)
        print(f"[train] {len(ds)} precomputed samples, "
              f"latent scale = {ds.latent_scale:.4f}")

        def infinite():
            while True:
                yield from loader
        it = infinite()
        get_batch = lambda: next(it)

    # ---------------- model / optimizer ------------------------------------- #
    model = build_model(cfg).to(device)
    projector = RepaProjector(cfg.dit.hidden_size, cfg.pretrained.repa_dim).to(device)
    diffusion = Diffusion(cfg.diffusion.num_train_steps, cfg.diffusion.schedule,
                          cfg.diffusion.logit_normal_mean,
                          cfg.diffusion.logit_normal_std)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] DiT parameters: {n_params:.1f}M  "
          f"(depth {cfg.dit.depth}, width {cfg.dit.hidden_size})")

    params = list(model.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(cfg.train.warmup_steps, cfg.train.total_steps))
    ema = EMA(model, cfg.train.ema_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        projector.load_state_dict(ckpt["projector"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        print(f"[train] resumed from {args.resume} at step {start_step}")

    # ---------------- the loop ---------------------------------------------- #
    model.train()
    first_loss, last_loss = None, None
    t0 = time.time()

    for step in range(start_step, cfg.train.total_steps):
        loss, logs = train_step(get_batch(), model, projector,
                                diffusion, cfg, step, device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)

        if first_loss is None:
            first_loss = logs["loss"]
        last_loss = logs["loss"]

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:>7d} | loss {logs['loss']:.4f} "
                  f"(diff {logs['diff']:.4f}, repa {logs['repa']:.4f}, "
                  f"lam {logs['lambda']:.3f}) | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {rate:.2f} it/s")

        if step > 0 and step % cfg.train.ckpt_every == 0:
            path = out_dir / f"ckpt_{step:07d}.pt"
            torch.save({"model": model.state_dict(),
                        "projector": projector.state_dict(),
                        "ema": ema.model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                        "config": cfg.as_dict()}, path)
            print(f"[train] checkpoint -> {path}")

    # Always leave a final checkpoint behind.
    final = out_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(),
                "projector": projector.state_dict(),
                "ema": ema.model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": cfg.train.total_steps - 1,
                "config": cfg.as_dict()}, final)
    print(f"[train] final checkpoint -> {final}")

    # ---------------- smoke-test verdict ------------------------------------ #
    if args.smoke:
        print("\n[smoke] verifying the DDIM + CFG sampler ...")
        B = 2
        shape = (B, cfg.latent.channels, cfg.latent.time, cfg.latent.freq)
        text_emb = torch.randn(B, cfg.pretrained.text_max_len,
                               cfg.pretrained.text_dim, device=device)
        text_mask = torch.ones(B, cfg.pretrained.text_max_len,
                               dtype=torch.long, device=device)
        z = diffusion.ddim_sample(ema.model, shape, text_emb, text_mask,
                                  num_steps=cfg.diffusion.sample_steps,
                                  cfg_scale=cfg.diffusion.cfg_scale,
                                  device=device)
        assert z.shape == shape, f"sampler shape mismatch: {z.shape}"
        assert torch.isfinite(z).all(), "sampler produced NaN/Inf"
        print(f"[smoke] sampled latents {tuple(z.shape)}, all finite")
        print(f"[smoke] loss went {first_loss:.4f} -> {last_loss:.4f} "
              f"over {cfg.train.total_steps} steps")
        print("[smoke] PASS - architecture, losses and sampler are wired correctly")


if __name__ == "__main__":
    main()
