"""
Diffusion Transformer (DiT) for text-to-audio latent diffusion - pure PyTorch.

Follows DiT (Peebles & Xie, 2023) with two additions needed for this project:

  1. CROSS-ATTENTION to T5 text tokens inside every block (PixArt-alpha style),
     so captions steer generation at token granularity.
  2. A "REPA tap": the forward pass can return the hidden state after an early
     block, which train.py aligns with frozen AST features (REPA loss).

Block layout (repeated `depth` times):

    x -> [adaLN-Zero LN -> Self-Attention  -> gate] -> +residual
      -> [        LN    -> Cross-Attention -> gate] -> +residual
      -> [adaLN-Zero LN -> MLP             -> gate] -> +residual

adaLN-Zero: the timestep (+ pooled text) embedding regresses per-block
(shift, scale, gate) triples.  All gates are initialised to ZERO, so at step 0
every block is the identity function - a major training-stability win that we
keep from the original DiT.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Embedders                                                                   #
# --------------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    """Map a scalar diffusion timestep t to a `hidden_size` vector.

    Classic sinusoidal features (like Transformer positions) followed by a
    2-layer MLP.  The output is the global conditioning vector that drives
    every adaLN-Zero modulation in the network.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        """t: [B] (float, any scale) -> [B, dim] sinusoidal features."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]                    # [B, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t, self.freq_dim))


def build_2d_sincos_pos_embed(dim: int, grid_t: int, grid_f: int) -> torch.Tensor:
    """Fixed 2-D sin/cos positional embedding over the (time, freq) token grid.

    The latent is spatial (time x frequency), so each axis gets half of the
    embedding dimensions.  Returns [grid_t * grid_f, dim], flattened
    time-major (token n = t * grid_f + f) to match PatchEmbed's flatten order.
    """
    assert dim % 4 == 0, "pos-embed dim must be divisible by 4 (2 axes x sin/cos)"

    def axis_embed(positions: torch.Tensor, axis_dim: int) -> torch.Tensor:
        omega = torch.arange(axis_dim // 2, dtype=torch.float32) / (axis_dim // 2)
        omega = 1.0 / (10_000 ** omega)                            # [axis_dim/2]
        out = positions.float()[:, None] * omega[None]             # [N, axis_dim/2]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [N, axis_dim]

    t_pos = torch.arange(grid_t).repeat_interleave(grid_f)         # 0,0,..,1,1,..
    f_pos = torch.arange(grid_f).repeat(grid_t)                    # 0,1,..,0,1,..
    emb = torch.cat([axis_embed(t_pos, dim // 2), axis_embed(f_pos, dim // 2)], dim=-1)
    return emb                                                     # [T*F, dim]


class PatchEmbed(nn.Module):
    """Latent [B, C, T, F] -> token sequence [B, N, hidden] via p x p patches.

    Implemented as a strided Conv2d (the standard ViT trick): each p x p patch
    of the latent becomes exactly one token.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # [B, hidden, T/p, F/p]
        return x.flatten(2).transpose(1, 2)    # [B, N, hidden]  (time-major)


# --------------------------------------------------------------------------- #
#  Attention primitives (hand-rolled, no external libs)                        #
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Standard multi-head self-attention over the latent tokens."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # One fused projection, then split into Q, K, V and the head dimension.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each [B, heads, N, head_dim]
        x = F.scaled_dot_product_attention(q, k, v)   # flash/mem-efficient kernel
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class CrossAttention(nn.Module):
    """Latent tokens attend to T5 text tokens.

    Queries come from the latent stream, keys/values from the (projected)
    text hidden states.  `text_mask` marks REAL tokens with 1 - padding
    positions are masked out of the attention so the model never conditions
    on pad garbage (a classic silent bug when forgotten).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor | None) -> torch.Tensor:
        B, N, D = x.shape
        L = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, L, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)       # each [B, heads, L, head_dim]

        attn_mask = None
        if ctx_mask is not None:
            # [B, L] {0,1} -> additive mask broadcast over heads and queries.
            attn_mask = torch.where(ctx_mask.bool(), 0.0, float("-inf"))
            attn_mask = attn_mask[:, None, None, :].to(q.dtype)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: LayerNorm output is shifted/scaled per sample."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------------------------- #
#  DiT block                                                                   #
# --------------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        # elementwise_affine=False: the affine part is provided by adaLN.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads)
        self.norm_ctx = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

        # adaLN-Zero head: regress (shift, scale, gate) for self-attn and MLP
        # from the conditioning vector -> 6 chunks of `dim`.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)   # ZERO init => block starts as
        nn.init.zeros_(self.adaLN[-1].bias)     # identity (adaLN-Zero).
        # Zero-init the cross-attention output too, so text conditioning fades
        # in smoothly instead of destabilising early training.
        nn.init.zeros_(self.cross.out.weight)
        nn.init.zeros_(self.cross.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                ctx: torch.Tensor, ctx_mask: torch.Tensor | None) -> torch.Tensor:
        (shift_sa, scale_sa, gate_sa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN(cond).chunk(6, dim=-1)

        # 1) self-attention (adaLN-Zero modulated + gated residual)
        x = x + gate_sa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_sa, scale_sa))
        # 2) cross-attention to text (plain pre-LN residual, zero-init output)
        x = x + self.cross(self.norm_ctx(x), ctx, ctx_mask)
        # 3) MLP (adaLN-Zero modulated + gated residual)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm -> linear projection back to patch pixels."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        # Zero init => the network initially predicts 0 everywhere, which is
        # the correct "do nothing" prior for a residual denoiser.
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# --------------------------------------------------------------------------- #
#  The full model                                                              #
# --------------------------------------------------------------------------- #
class DiT(nn.Module):
    """Text-conditioned Diffusion Transformer over VAE mel-latents.

    forward(z_t, t, text_emb, text_mask) -> prediction [B, C, T, F]
    (the prediction target - epsilon or v - is chosen by the diffusion code,
    the network is agnostic).
    """

    def __init__(self, latent_channels: int, latent_time: int, latent_freq: int,
                 patch_size: int, hidden_size: int, depth: int, num_heads: int,
                 mlp_ratio: float, text_dim: int, repa_layer: int):
        super().__init__()
        assert latent_time % patch_size == 0 and latent_freq % patch_size == 0
        self.out_channels = latent_channels
        self.patch_size = patch_size
        self.grid_t = latent_time // patch_size
        self.grid_f = latent_freq // patch_size
        self.num_tokens = self.grid_t * self.grid_f
        self.repa_layer = repa_layer

        self.patch_embed = PatchEmbed(latent_channels, hidden_size, patch_size)
        # Fixed (non-trained) 2-D sincos positional embedding.
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(hidden_size, self.grid_t, self.grid_f)[None],
            persistent=False,
        )

        self.t_embed = TimestepEmbedder(hidden_size)
        # Trainable projection: T5 hidden size -> DiT width (the "glue" layer).
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_size),
        )
        # Pooled-text -> conditioning vector: gives every adaLN a global text
        # signal in addition to the per-token cross-attention.
        self.pooled_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Learned NULL text sequence for classifier-free guidance (one token).
        # Training replaces the caption with this with prob p_uncond; sampling
        # uses it as the unconditional branch of CFG.
        self.null_text = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        # Xavier for generic linears; the zero-inits inside DiTBlock/FinalLayer
        # were applied at construction and are re-applied after this pass.
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic)
        # Patchify conv: treat as a linear layer over flattened patches.
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:           # restore the adaLN-Zero contract
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
            nn.init.zeros_(block.cross.out.weight)
            nn.init.zeros_(block.cross.out.bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    # ------------------------------------------------------------------ #
    def null_context(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The learned unconditional context (already in DiT width) + mask."""
        ctx = self.null_text.expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, 1, device=ctx.device, dtype=torch.long)
        return ctx, mask

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, p*p*C] tokens -> [B, C, T, F] latent grid."""
        B = x.shape[0]
        p, c = self.patch_size, self.out_channels
        x = x.reshape(B, self.grid_t, self.grid_f, p, p, c)
        x = torch.einsum("btfpqc->bctpfq", x)           # interleave patch pixels
        return x.reshape(B, c, self.grid_t * p, self.grid_f * p)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                text_emb: torch.Tensor | None, text_mask: torch.Tensor | None,
                drop_mask: torch.Tensor | None = None,
                return_repa_hidden: bool = False):
        """
        z_t        : [B, C, T, F]  noisy latent
        t          : [B]           diffusion timestep (float in [0, T))
        text_emb   : [B, L, text_dim] T5 hidden states (None => unconditional)
        text_mask  : [B, L]        1 = real token, 0 = padding
        drop_mask  : [B] bool      True => replace this sample's caption with
                                   the null context (CFG dropout, train only)
        """
        B = z_t.shape[0]
        x = self.patch_embed(z_t) + self.pos_embed      # [B, N, hidden]

        # --- build the text context in DiT width -------------------------- #
        if text_emb is None:
            ctx, ctx_mask = self.null_context(B)
        else:
            ctx = self.text_proj(text_emb)              # [B, L, hidden]
            ctx_mask = text_mask
            # NOTE: no `.any()` short-circuit here - reading a device tensor
            # from Python forces a host<->device sync every step, which stalls
            # XLA/TPU pipelines.  torch.where with an all-False mask is free.
            if drop_mask is not None:
                # Per-sample CFG dropout: pad the null token out to length L so
                # conditional/unconditional samples share one tensor shape.
                null = self.null_text.expand(B, ctx.shape[1], -1)
                ctx = torch.where(drop_mask[:, None, None], null, ctx)
                null_mask = torch.zeros_like(ctx_mask)
                null_mask[:, 0] = 1                     # only token 0 is "real"
                ctx_mask = torch.where(drop_mask[:, None], null_mask, ctx_mask)

        # --- global conditioning vector (timestep + pooled text) ---------- #
        cond = self.t_embed(t)
        if ctx_mask is not None:
            denom = ctx_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (ctx * ctx_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = ctx.mean(dim=1)
        cond = cond + self.pooled_proj(pooled)

        # --- transformer trunk with optional REPA tap ---------------------- #
        repa_hidden = None
        for i, block in enumerate(self.blocks):
            x = block(x, cond, ctx, ctx_mask)
            if return_repa_hidden and i == self.repa_layer:
                repa_hidden = x                          # [B, N, hidden]

        out = self.unpatchify(self.final(x, cond))       # [B, C, T, F]
        if return_repa_hidden:
            return out, repa_hidden
        return out


# --------------------------------------------------------------------------- #
#  REPA projection head (trainable, train-time only, dropped at inference)     #
# --------------------------------------------------------------------------- #
class RepaProjector(nn.Module):
    """3-layer MLP mapping DiT hidden tokens -> AST feature space.

    REPA aligns g(h_l) with the frozen encoder's features of the CLEAN input
    via a per-token cosine loss.  This head (and the loss) exist only during
    training; sampling never touches them.
    """

    def __init__(self, dit_dim: int, target_dim: int, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dit_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def repa_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative mean per-token cosine similarity.

    projected : [B, N, D] = g(h_l)   (trainable path)
    target    : [B, N, D] = y*       (frozen AST features, precomputed on the
                                      same N-token grid as the DiT)
    """
    projected = F.normalize(projected, dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(projected * target).sum(dim=-1).mean()
