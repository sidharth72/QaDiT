"""
Diffusion process - schedules, forward noising, v-prediction, DDIM sampling.

Everything here is plain tensor math (no diffusers).  We use:

  * a COSINE alpha-bar schedule (Nichol & Dhariwal, 2021),
  * V-PREDICTION as the network target (Salimans & Ho, 2022):
        v = sqrt(abar_t) * eps - sqrt(1 - abar_t) * z0
    which trains more stably than raw epsilon for transformer backbones and
    interpolates between predicting noise (t small) and data (t large),
  * LOGIT-NORMAL timestep sampling at train time (SD3): more probability mass
    on mid-range noise levels where the model actually learns,
  * a DDIM sampler with CLASSIFIER-FREE GUIDANCE for inference.

Key identities used throughout (see DDPM.md notes):
    z_t  = sqrt(abar_t) * z0 + sqrt(1 - abar_t) * eps      (forward jump)
    z0   = sqrt(abar_t) * z_t - sqrt(1 - abar_t) * v       (recover data)
    eps  = sqrt(1 - abar_t) * z_t + sqrt(abar_t) * v       (recover noise)
"""

import torch
import math

class Diffusion:
    """Holds the schedule tables and implements q(z_t|z0), targets, sampling."""

    def __init__(self, num_train_steps: int = 1000, schedule: str = "cosine",
                 logit_normal_mean: float = 0.0, logit_normal_std: float = 1.0):
        self.T = num_train_steps
        self.ln_mean = logit_normal_mean
        self.ln_std = logit_normal_std

        if schedule == "cosine":
            # abar(t) = cos^2( (t/T + s) / (1 + s) * pi/2 ), s = 0.008.
            # Clipped so abar never reaches exactly 0 or 1 (numerical safety).
            s = 0.008
            steps = torch.arange(self.T + 1, dtype=torch.float64)
            f = torch.cos((steps / self.T + s) / (1 + s) * math.pi / 2) ** 2
            abar = (f / f[0]).clamp(1e-5, 1.0)
            self.alpha_bar = abar[1:].float()          # abar at t = 1..T -> index t-1
        else:
            raise ValueError(f"unknown schedule: {schedule}")

    def to(self, device) -> "Diffusion":
        """Move the schedule table to the accelerator ONCE.

        On XLA, leaving alpha_bar on CPU would re-upload it (host->device
        transfer) inside every training step; pinning it on the device keeps
        the step graph free of host traffic.
        """
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    # ------------------------------------------------------------------ #
    #  Schedule lookups                                                    #
    # ------------------------------------------------------------------ #
    def _gather(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """abar coefficients for integer timesteps t in [0, T), broadcastable
        against a [B, C, T, F] latent.  Returns (sqrt_abar, sqrt_one_minus)."""
        abar = self.alpha_bar.to(t.device)[t]          # [B]
        sqrt_abar = abar.sqrt().view(-1, 1, 1, 1)
        sqrt_1m = (1 - abar).sqrt().view(-1, 1, 1, 1)
        return sqrt_abar, sqrt_1m

    # ------------------------------------------------------------------ #
    #  Training utilities                                                  #
    # ------------------------------------------------------------------ #
    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        """Logit-normal timestep sampling.

        u ~ N(mean, std); sigmoid(u) in (0,1) is mapped to an integer step.
        Compared to uniform sampling this concentrates training on mid-noise
        timesteps, which measurably speeds up DiT convergence.
        """
        u = torch.randn(batch_size, device=device) * self.ln_std + self.ln_mean
        frac = torch.sigmoid(u)
        return (frac * self.T).long().clamp(0, self.T - 1)

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor,
                  eps: torch.Tensor) -> torch.Tensor:
        """Forward process q(z_t | z0): jump straight to noise level t."""
        sqrt_abar, sqrt_1m = self._gather(t)
        return sqrt_abar * z0 + sqrt_1m * eps

    def v_target(self, z0: torch.Tensor, t: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        """The v-prediction regression target."""
        sqrt_abar, sqrt_1m = self._gather(t)
        return sqrt_abar * eps - sqrt_1m * z0

    # ------------------------------------------------------------------ #
    #  Conversions (used by the sampler)                                   #
    # ------------------------------------------------------------------ #
    def z0_from_v(self, z_t: torch.Tensor, t: torch.Tensor,
                  v: torch.Tensor) -> torch.Tensor:
        sqrt_abar, sqrt_1m = self._gather(t)
        return sqrt_abar * z_t - sqrt_1m * v

    def eps_from_v(self, z_t: torch.Tensor, t: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
        sqrt_abar, sqrt_1m = self._gather(t)
        return sqrt_1m * z_t + sqrt_abar * v

    # ------------------------------------------------------------------ #
    #  DDIM sampling with classifier-free guidance                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ddim_sample(self, model, shape: tuple, text_emb: torch.Tensor,
                    text_mask: torch.Tensor, num_steps: int = 50,
                    cfg_scale: float = 4.0, eta: float = 0.0,
                    device: str = "cpu",
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate latents from pure noise, guided by text.

        model     : the (EMA) DiT - called twice per step (cond + uncond) when
                    cfg_scale > 1, batched together for efficiency.
        shape     : [B, C, T, F] latent shape to generate.
        eta       : 0.0 = deterministic DDIM; 1.0 = ancestral (DDPM-like).
        Returns z0-hat in the SCALED latent space (divide by the latent scale
        factor before VAE decoding - sample.py handles that).
        """
        B = shape[0]
        z = torch.randn(shape, device=device, generator=generator)

        # Evenly spaced timestep subsequence T-1 ... 0 (e.g. 50 of 1000 steps).
        times = torch.linspace(self.T - 1, 0, num_steps, device=device).long()

        # CFG needs a conditional and an unconditional prediction per step.
        # The unconditional branch uses the model's learned null context
        # (text_emb=None path).  We keep the two passes separate for clarity;
        # batching them into one 2B forward is a straightforward optimisation.
        use_cfg = cfg_scale is not None and cfg_scale > 1.0

        for i in range(num_steps):
            t = times[i].expand(B)                       # current step, [B]

            if use_cfg:
                # Two predictions: with caption and with the learned null.
                v_cond = model(z, t.float(), text_emb, text_mask)
                v_uncond = model(z, t.float(), None, None)
                # CFG extrapolation in v-space (linear in the prediction):
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(z, t.float(), text_emb, text_mask)

            # Convert the v prediction into (z0-hat, eps-hat) at this level.
            z0_hat = self.z0_from_v(z, t, v)
            eps_hat = self.eps_from_v(z, t, v)

            if i == num_steps - 1:
                z = z0_hat                               # final step: output data
                break

            # DDIM update to the next (lower) noise level.
            t_next = times[i + 1].expand(B)
            abar_next = self.alpha_bar.to(device)[t_next].view(-1, 1, 1, 1)
            abar_now = self.alpha_bar.to(device)[t].view(-1, 1, 1, 1)

            # Optional stochasticity (eta > 0 recovers DDPM-like sampling).
            sigma = eta * torch.sqrt((1 - abar_next) / (1 - abar_now)
                                     * (1 - abar_now / abar_next))
            noise = torch.randn(shape, device=device, generator=generator) \
                if eta > 0 else torch.zeros_like(z)

            dir_zt = torch.sqrt((1 - abar_next - sigma ** 2).clamp(min=0.0)) * eps_hat
            z = abar_next.sqrt() * z0_hat + dir_zt + sigma * noise

        return z
