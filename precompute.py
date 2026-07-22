"""
Offline pre-compute stage: run every FROZEN pretrained model exactly once per
clip and cache the results to disk, so the training loop only ever loads small
tensors and runs the DiT (the single biggest efficiency win of the project).

For every AudioCaps clip this script stores:

  latent     [8, 256, 16]   AudioLDM VAE posterior mean of the log-mel
                            (RAW, unscaled - the scale lives in meta.json)
  text_emb   [64, 1024]     FLAN-T5-large encoder hidden states (fp16)
  text_mask  [64]           1 = real token, 0 = padding
  repa       [1024, 768]    AST features of the CLEAN audio, bilinearly
                            resampled onto the DiT's 1024-token (128 x 8)
                            grid so train.py can apply the cosine loss with
                            zero reshaping (fp16)
  caption    str            kept for logging / eyeballing samples

Output layout (one folder per split):

  <out>/train/shard_00000.pt   { "latent": [N,...], "text_emb": ..., ... }
  <out>/train/meta.json        latent scale, shapes, config snapshot

Usage:
  python precompute.py --out ./cache --split train
  python precompute.py --out ./cache --split train --max-samples 512   # smoke
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import Config


# --------------------------------------------------------------------------- #
#  Mel front-end (HiFi-GAN / AudioLDM style - the "locked triple" contract)   #
# --------------------------------------------------------------------------- #
class LogMel:
    """waveform [B, num_samples] -> log-mel [B, 1, frames, n_mels].

    Matches the HiFi-GAN mel recipe AudioLDM's VAE/vocoder were trained on:
    reflect-padded, center=False STFT; slaney-normalised mel filterbank;
    natural log with a 1e-5 floor.  Time is the image HEIGHT, mel bins the
    WIDTH - that is the orientation the AudioLDM VAE expects.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.window = torch.hann_window(cfg.win_length, device=device)
        # Slaney-style filterbank = what librosa (and hence AudioLDM) uses.
        self.mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=cfg.n_fft // 2 + 1,
            f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            norm="slaney", mel_scale="slaney",
        ).to(device)                                     # [n_freqs, n_mels]

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # HiFi-GAN pads so that frame count == num_samples / hop exactly.
        pad = (cfg.n_fft - cfg.hop_length) // 2
        wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                          win_length=cfg.win_length, window=self.window,
                          center=False, return_complex=True)
        mag = spec.abs()                                 # [B, n_freqs, frames]
        mel = mag.transpose(1, 2) @ self.mel_fb          # [B, frames, n_mels]
        logmel = torch.log(mel.clamp(min=1e-5))
        return logmel.unsqueeze(1)                       # [B, 1, frames, n_mels]


def fix_length(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Zero-pad or trim a mono waveform to exactly `num_samples`."""
    if wav.shape[-1] >= num_samples:
        return wav[..., :num_samples]
    return F.pad(wav, (0, num_samples - wav.shape[-1]))


def decode_audio(example_audio, target_sr: int) -> torch.Tensor:
    """Robustly pull a mono float32 waveform out of a HF `Audio` cell and
    resample it to `target_sr`.  Handles both the classic dict format
    (datasets < 4) and torchcodec AudioDecoder objects (datasets >= 4)."""
    if isinstance(example_audio, dict):                  # datasets < 4.x
        arr = np.asarray(example_audio["array"], dtype=np.float32)
        sr = int(example_audio["sampling_rate"])
    else:                                                # torchcodec decoder
        samples = example_audio.get_all_samples()
        arr = samples.data.numpy().astype(np.float32)
        sr = int(samples.sample_rate)
    wav = torch.from_numpy(arr)
    if wav.ndim == 2:                                    # [channels, T] -> mono
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


# --------------------------------------------------------------------------- #
#  AST feature regridding                                                      #
# --------------------------------------------------------------------------- #
def regrid_ast_features(hidden: torch.Tensor, ast_grid: tuple[int, int],
                        dit_grid: tuple[int, int]) -> torch.Tensor:
    """Resample AST patch tokens onto the DiT token grid.

    hidden   : [B, 2 + T_ast*F_ast, 768]  (AST prepends CLS + distill tokens)
    ast_grid : (time_patches, freq_patches) of AST's patchifier
    dit_grid : (grid_t, grid_f) of the DiT = (128, 8)

    Both models patchify a (time x freq) plane, so a bilinear resize on the
    token grid gives a semantically aligned per-token target y*.
    Returns [B, grid_t * grid_f, 768], flattened time-major like the DiT.
    """
    B, _, D = hidden.shape
    t_ast, f_ast = ast_grid
    patches = hidden[:, 2:, :]                           # drop CLS + distill
    patches = patches.reshape(B, t_ast, f_ast, D).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=dit_grid,
                            mode="bilinear", align_corners=False)
    return patches.permute(0, 2, 3, 1).reshape(B, dit_grid[0] * dit_grid[1], D)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./cache",
                    help="output root; a subfolder per split is created")
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "validation", "test"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="stop early (useful for smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="clips per forward pass through the frozen models")
    ap.add_argument("--shard-size", type=int, default=256,
                    help="samples per output .pt shard")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] split={args.split}  device={device}  out={out_dir}")

    # ---------------- frozen pretrained models (loaded once) -------------- #
    # Heavy imports are local so `--help` stays instant.
    from datasets import load_dataset
    from diffusers import AutoencoderKL
    from transformers import (ASTFeatureExtractor, ASTModel,
                              AutoTokenizer, T5EncoderModel)

    print("[precompute] loading AudioLDM VAE ...")
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained.vae_repo, subfolder="vae").to(device).eval()

    print("[precompute] loading FLAN-T5 ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained.text_model)
    t5 = T5EncoderModel.from_pretrained(cfg.pretrained.text_model).to(device).eval()

    print("[precompute] loading AST (REPA target encoder) ...")
    ast_extractor = ASTFeatureExtractor.from_pretrained(cfg.pretrained.repa_model)
    ast = ASTModel.from_pretrained(cfg.pretrained.repa_model).to(device).eval()
    # AST patchifies its (1024 frames x 128 mel) fbank input with 16x16
    # kernels at stride 10 -> a (101 time x 12 freq) patch grid.
    ast_grid = (
        (ast.config.max_length - ast.config.patch_size) // ast.config.time_stride + 1,       # time: 101
        (ast.config.num_mel_bins - ast.config.patch_size) // ast.config.frequency_stride + 1, # freq: 12
    )
    print(f"[precompute] AST patch grid (time, freq) = {ast_grid}")

    logmel = LogMel(cfg.mel, device)
    dit_grid = (cfg.latent.time // cfg.dit.patch_size,
                cfg.latent.freq // cfg.dit.patch_size)   # (128, 8)

    # ---------------- dataset --------------------------------------------- #
    print("[precompute] loading OpenSound/AudioCaps ...")
    ds = load_dataset("OpenSound/AudioCaps", split=args.split)
    n_total = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    print(f"[precompute] processing {n_total} / {len(ds)} clips")

    # Running stats for the latent scale factor (unit-variance trick, see
    # AudioDiffusionModel.md section 5).  Computed over every latent element.
    running_sum, running_sumsq, running_count = 0.0, 0.0, 0

    shard: dict[str, list] = {"latent": [], "text_emb": [], "text_mask": [],
                              "repa": [], "caption": []}
    shard_idx, in_shard = 0, 0
    shard_files: list[str] = []

    def flush_shard():
        nonlocal shard, shard_idx, in_shard
        if in_shard == 0:
            return
        path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({
            "latent": torch.stack(shard["latent"]),          # [N, 8, 256, 16] fp16
            "text_emb": torch.stack(shard["text_emb"]),      # [N, 64, 1024]  fp16
            "text_mask": torch.stack(shard["text_mask"]),    # [N, 64]        uint8
            "repa": torch.stack(shard["repa"]),              # [N, 1024, 768] fp16
            "caption": shard["caption"],
        }, path)
        print(f"[precompute] wrote {path.name}  ({in_shard} samples)")
        shard_files.append(path.name)
        shard = {k: [] for k in shard}
        shard_idx += 1
        in_shard = 0

    # ---------------- the loop -------------------------------------------- #
    for start in range(0, n_total, args.batch_size):
        rows = ds[start:min(start + args.batch_size, n_total)]
        captions = rows["caption"]

        # -- audio -> fixed-length 16 kHz mono batch ----------------------- #
        wavs = torch.stack([
            fix_length(decode_audio(a, cfg.mel.sample_rate), cfg.mel.num_samples)
            for a in rows["audio"]
        ]).to(device)                                        # [B, 163840]

        # -- 1) mel -> VAE latent (posterior mean, raw/unscaled) ----------- #
        mel = logmel(wavs)                                   # [B, 1, 1024, 64]
        latents = vae.encode(mel).latent_dist.mode()         # [B, 8, 256, 16]

        running_sum += latents.double().sum().item()
        running_sumsq += (latents.double() ** 2).sum().item()
        running_count += latents.numel()

        # -- 2) caption -> T5 hidden states + mask ------------------------- #
        tok = tokenizer(captions, padding="max_length", truncation=True,
                        max_length=cfg.pretrained.text_max_len,
                        return_tensors="pt").to(device)
        text_emb = t5(input_ids=tok.input_ids,
                      attention_mask=tok.attention_mask).last_hidden_state

        # -- 3) clean audio -> AST features on the DiT token grid ---------- #
        # ASTFeatureExtractor wants raw 16 kHz waveforms (numpy, on CPU); it
        # builds the 128-bin fbank AST was trained on internally.
        ast_inputs = ast_extractor(
            [w.cpu().numpy() for w in wavs],
            sampling_rate=cfg.mel.sample_rate, return_tensors="pt",
        ).to(device)
        ast_hidden = ast(**ast_inputs).last_hidden_state     # [B, 1214, 768]
        repa = regrid_ast_features(ast_hidden, ast_grid, dit_grid)

        # -- stash into the current shard ----------------------------------- #
        for i in range(latents.shape[0]):
            shard["latent"].append(latents[i].half().cpu())
            shard["text_emb"].append(text_emb[i].half().cpu())
            shard["text_mask"].append(tok.attention_mask[i].to(torch.uint8).cpu())
            shard["repa"].append(repa[i].half().cpu())
            shard["caption"].append(captions[i])
            in_shard += 1
            if in_shard >= args.shard_size:
                flush_shard()

        done = min(start + args.batch_size, n_total)
        if (start // args.batch_size) % 10 == 0:
            print(f"[precompute] {done}/{n_total} clips")

    flush_shard()

    # ---------------- meta.json ------------------------------------------- #
    mean = running_sum / running_count
    std = math.sqrt(max(running_sumsq / running_count - mean ** 2, 1e-12))
    meta = {
        "split": args.split,
        "num_samples": n_total,
        "shards": shard_files,
        # Multiply latents by this to get ~unit variance (divide back before
        # VAE decode).  This is the Stable-Diffusion-0.18215 analog, estimated
        # from this dataset instead of hard-coded.
        "latent_scale": 1.0 / std,
        "latent_mean": mean,
        "latent_std": std,
        "vae_scaling_factor": getattr(vae.config, "scaling_factor", None),
        "shapes": {
            "latent": [cfg.latent.channels, cfg.latent.time, cfg.latent.freq],
            "text_emb": [cfg.pretrained.text_max_len, cfg.pretrained.text_dim],
            "repa": [dit_grid[0] * dit_grid[1], cfg.pretrained.repa_dim],
        },
        "config": cfg.as_dict(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[precompute] done.  latent std = {std:.4f} -> scale = {1.0/std:.4f}")
    print(f"[precompute] meta written to {out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
