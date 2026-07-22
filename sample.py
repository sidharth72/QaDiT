"""
Inference: caption -> waveform.

Chain (see AudioDiffusionModel.md section 16):

    caption -> FLAN-T5 -> c
    z_T ~ N(0, I) -> [ DDIM reverse steps with CFG, EMA DiT ] -> z0'-hat
    z0'-hat -> divide by latent_scale -> VAE.decode -> mel -> HiFi-GAN -> wav

The frozen models (T5, VAE, vocoder) are loaded here because inference needs
them live; training never does (it reads the precomputed cache).

Usage:
  python sample.py --ckpt runs/dit_b2/ckpt_final.pt --cache ./cache \
      --prompt "A dog barks while birds chirp in the distance" \
      --out dog_birds.wav
"""

import argparse
import json
from pathlib import Path

import torch

from config import Config
from diffusion import Diffusion
from train import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="checkpoint from train.py (the EMA weights are used)")
    ap.add_argument("--cache", type=str, default="./cache",
                    help="precompute cache root (for meta.json latent_scale)")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default="sample.wav")
    ap.add_argument("--steps", type=int, default=None, help="DDIM steps")
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    steps = args.steps or cfg.diffusion.sample_steps
    cfg_scale = args.cfg_scale or cfg.diffusion.cfg_scale

    # ---------------- latent scale (must match training data) -------------- #
    meta_path = Path(args.cache) / "train" / "meta.json"
    with open(meta_path) as f:
        latent_scale = float(json.load(f)["latent_scale"])
    print(f"[sample] latent scale = {latent_scale:.4f}")

    # ---------------- EMA DiT ---------------------------------------------- #
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["ema"])       # ALWAYS sample from the EMA copy
    print(f"[sample] loaded EMA weights from {args.ckpt} (step {ckpt['step']})")

    # ---------------- frozen helpers --------------------------------------- #
    from diffusers import AutoencoderKL
    from transformers import AutoTokenizer, SpeechT5HifiGan, T5EncoderModel
    import soundfile as sf

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained.text_model)
    t5 = T5EncoderModel.from_pretrained(cfg.pretrained.text_model).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained.vae_repo, subfolder="vae").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(
        cfg.pretrained.vocoder_repo, subfolder="vocoder").to(device).eval()

    # ---------------- 1) caption -> T5 tokens ------------------------------- #
    tok = tokenizer([args.prompt], padding="max_length", truncation=True,
                    max_length=cfg.pretrained.text_max_len,
                    return_tensors="pt").to(device)
    text_emb = t5(input_ids=tok.input_ids,
                  attention_mask=tok.attention_mask).last_hidden_state
    text_mask = tok.attention_mask

    # ---------------- 2) reverse diffusion with CFG -------------------------- #
    diffusion = Diffusion(cfg.diffusion.num_train_steps, cfg.diffusion.schedule)
    gen = torch.Generator(device=device.type).manual_seed(args.seed)
    shape = (1, cfg.latent.channels, cfg.latent.time, cfg.latent.freq)
    print(f"[sample] sampling {steps} DDIM steps, cfg_scale={cfg_scale} ...")
    z = diffusion.ddim_sample(model, shape, text_emb, text_mask,
                              num_steps=steps, cfg_scale=cfg_scale,
                              device=device, generator=gen)

    # ---------------- 3) latent -> mel -> waveform --------------------------- #
    z = z / latent_scale                       # undo the unit-variance scaling
    mel = vae.decode(z).sample                 # [1, 1, 1024, 64] log-mel
    # SpeechT5HifiGan wants [B, frames, n_mels]
    wav = vocoder(mel.squeeze(1))              # [1, num_samples]
    wav = wav.squeeze().clamp(-1, 1).cpu().numpy()

    sf.write(args.out, wav, cfg.mel.sample_rate)
    dur = len(wav) / cfg.mel.sample_rate
    print(f"[sample] wrote {args.out}  ({dur:.2f} s @ {cfg.mel.sample_rate} Hz)")
    print(f"[sample] prompt: {args.prompt!r}")


if __name__ == "__main__":
    main()
