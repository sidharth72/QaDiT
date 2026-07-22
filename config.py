"""
Central configuration for the text-to-audio latent DiT.

Everything that must stay CONSISTENT between precompute.py, train.py and
sample.py lives here.  The three stages share one source of truth so the
"locked triple" (mel config <-> VAE <-> vocoder) can never silently drift.

Shape conventions used throughout the project
---------------------------------------------
waveform : [B, 163840]           10.24 s of 16 kHz mono audio
mel      : [B, 1, 1024, 64]      (channel, TIME frames, MEL bins) - this is the
                                 orientation the AudioLDM VAE expects (time is
                                 the "image height", mel bins the "width")
latent   : [B, 8, 256, 16]       VAE downsamples time and freq by 4x
DiT grid : (128, 8) = 1024 tokens after 2x2 patchify (time-major flatten:
                                 token n = t * 8 + f)
text     : [B, 64, 1024]         FLAN-T5-large hidden states + [B, 64] mask
REPA y*  : [B, 1024, 768]        AST features bilinearly resampled onto the
                                 exact same 1024-token DiT grid at precompute
"""

from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------- #
#  Audio front-end  (LOCKED to the AudioLDM VAE + HiFi-GAN vocoder pair)      #
# --------------------------------------------------------------------------- #
@dataclass
class MelConfig:
    sample_rate: int = 16_000
    duration_s: float = 10.24          # 10.24 s -> exactly 1024 mel frames
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 160              # 10 ms hop at 16 kHz
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8_000.0

    @property
    def num_samples(self) -> int:      # 163_840 samples per clip
        return int(self.sample_rate * self.duration_s)

    @property
    def num_frames(self) -> int:       # 1024 mel frames per clip
        return self.num_samples // self.hop_length


# --------------------------------------------------------------------------- #
#  Frozen pretrained components (HuggingFace ids)                              #
# --------------------------------------------------------------------------- #
@dataclass
class PretrainedConfig:
    # AudioLDM ships a matched (VAE, vocoder) pair for the mel config above.
    vae_repo: str = "cvssp/audioldm-s-full-v2"     # subfolder="vae"
    vocoder_repo: str = "cvssp/audioldm-s-full-v2" # subfolder="vocoder"
    # Text encoder: token-level embeddings for cross-attention (Tango recipe).
    text_model: str = "google/flan-t5-large"       # hidden size 1024
    text_max_len: int = 64
    text_dim: int = 1024
    # REPA target: AST = Audio Spectrogram Transformer (audio analog of the
    # image SSL encoders REPA was designed around).  Hidden size 768.
    repa_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    repa_dim: int = 768


# --------------------------------------------------------------------------- #
#  Latent geometry (determined by the VAE, must match precompute output)      #
# --------------------------------------------------------------------------- #
@dataclass
class LatentConfig:
    channels: int = 8      # AudioLDM KL-VAE latent channels
    time: int = 256        # 1024 mel frames / 4x VAE downsample
    freq: int = 16         # 64 mel bins    / 4x VAE downsample


# --------------------------------------------------------------------------- #
#  DiT backbone                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class DiTConfig:
    patch_size: int = 2      # 2x2 patches over the (256, 16) latent -> 1024 tokens
    hidden_size: int = 768   # DiT-B width
    depth: int = 12          # DiT-B depth
    num_heads: int = 12
    mlp_ratio: float = 4.0
    # REPA taps the hidden state after this block (~depth/3, per the paper).
    repa_layer: int = 4


# --------------------------------------------------------------------------- #
#  Diffusion process                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class DiffusionConfig:
    num_train_steps: int = 1000     # discrete timesteps T
    schedule: str = "cosine"        # cosine alpha-bar schedule
    # Training-time timestep sampling: logit-normal concentrates samples in the
    # informative mid-noise regime (SD3 trick) instead of uniform.
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    # Classifier-free guidance
    p_uncond: float = 0.1           # caption dropout probability at train time
    cfg_scale: float = 4.0          # default guidance weight at sampling time
    sample_steps: int = 50          # DDIM steps at inference


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    total_steps: int = 200_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    # REPA loss weight: starts at repa_weight, linearly decays to 0 over
    # repa_decay_steps (REPA helps most early; let the diffusion loss own the
    # end of training).  Set repa_weight=0.0 to disable REPA entirely (A/B).
    repa_weight: float = 0.5
    repa_decay_steps: int = 200_000
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0


# --------------------------------------------------------------------------- #
#  Top-level bundle                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    mel: MelConfig = field(default_factory=MelConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ #
    # A tiny configuration whose forward pass runs in seconds on CPU.     #
    # Used by `train.py --smoke` to verify the whole pipeline end-to-end  #
    # (shapes, losses, sampler) without any real data or pretrained nets. #
    # ------------------------------------------------------------------ #
    @classmethod
    def smoke(cls) -> "Config":
        cfg = cls()
        cfg.latent = LatentConfig(channels=4, time=32, freq=8)
        cfg.dit = DiTConfig(patch_size=2, hidden_size=64, depth=3,
                            num_heads=4, mlp_ratio=2.0, repa_layer=1)
        cfg.pretrained.text_dim = 32
        cfg.pretrained.text_max_len = 8
        cfg.pretrained.repa_dim = 16
        cfg.diffusion.sample_steps = 5
        cfg.train.batch_size = 4
        cfg.train.warmup_steps = 2
        cfg.train.total_steps = 20
        cfg.train.repa_decay_steps = 20
        cfg.train.log_every = 1
        cfg.train.ckpt_every = 10
        return cfg
