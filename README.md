# audio_dit — Text-to-Audio Latent Diffusion (pure-PyTorch DiT + T5 + REPA)

PyTorch implementation of the architecture in
[`../AudioDiffusionModel.md`](../AudioDiffusionModel.md) /
[`../ArchitectureFlow.md`](../ArchitectureFlow.md).
The DiT, diffusion math and sampler are written from scratch (no `diffusers`
pipelines); HuggingFace libraries are used **only** to load the frozen
pretrained components (AudioLDM VAE, FLAN-T5, AST, HiFi-GAN vocoder).

## Files

| File | What it does |
|---|---|
| `config.py` | Single source of truth for every shape/hyperparameter shared across stages |
| `dit.py` | The DiT: patchify, 2-D sincos pos-emb, adaLN-Zero blocks, T5 cross-attention, REPA tap + projector |
| `diffusion.py` | Cosine schedule, forward noising, v-prediction targets, logit-normal `t` sampling, DDIM sampler with CFG |
| `precompute.py` | AudioCaps → cached VAE latents, T5 embeddings + masks, AST REPA targets (sharded `.pt` files + `meta.json`) |
| `dataset.py` | `Dataset`/collate over the precomputed shards (returns latents pre-scaled to unit variance) |
| `train.py` | Training loop (GPU/CPU): CFG dropout, REPA loss with decaying weight, EMA, warmup+cosine LR, checkpointing, `--smoke` mode |
| `train_xla.py` | The same training on TPU via PyTorch/XLA: 8-way data parallel, `MpDeviceLoader`, bf16 autocast, sync-free logging, `xm.save` checkpoints |
| `train.ipynb` | Kaggle launcher notebook for `train_xla.py` on a TPU v5e-8 |
| `sample.py` | Inference: caption → T5 → DDIM+CFG → VAE decode → HiFi-GAN → `.wav` |

## Install

```bash
pip install -r requirements.txt
```

(For the smoke test alone, `torch` is enough.)

## 1. Smoke test (no data, no downloads, CPU-friendly)

```bash
python train.py --smoke
```

Builds a tiny DiT, trains 20 steps on synthetic tensors with the exact real
shapes, then runs the DDIM+CFG sampler. Expect `[smoke] PASS` at the end —
that means the model, both losses, EMA and the sampler are all wired correctly.

## 2. Precompute (run once per split)

```bash
python precompute.py --out ./cache --split train
python precompute.py --out ./cache --split validation
```

Downloads `OpenSound/AudioCaps` via HF `datasets`, and for each clip caches:
VAE latent `[8,256,16]`, T5 hidden states `[64,1024]` + mask, and AST REPA
targets `[1024,768]` already resampled onto the DiT token grid. fp16 on disk,
~2 MB/clip → **~90 GB for the full 45k train split**. For a first GPU run try
`--max-samples 2000`.

## 3. Train

```bash
python train.py --data ./cache --out ./runs/dit_b2
```

Resume after an interruption with `--resume runs/dit_b2/ckpt_0002000.pt`.
Disable REPA for an A/B by setting `repa_weight = 0.0` in `config.py`.

## 4. Generate audio

```bash
python sample.py --ckpt runs/dit_b2/ckpt_final.pt --cache ./cache \
    --prompt "A dog barks while birds chirp in the distance" --out dog.wav
```

## Sanity check the frozen triple first

Before burning GPU-hours, verify the VAE + vocoder round-trip on a few real
clips (`wav → mel → VAE encode → decode → vocoder → wav`) and *listen* to the
result — it is the quality ceiling of everything downstream
(see `AudioDiffusionModel.md` §6).

## 5. Train on Kaggle TPU v5e-8 (PyTorch/XLA)

Upload the `.py` files as one Kaggle Dataset and the precompute cache as
another, open `train.ipynb` with the TPU v5e-8 accelerator, and run it — it
stages the code and launches:

```bash
python train_xla.py --data /kaggle/input/audiocaps-precomputed \
    --out /kaggle/working/runs/dit_b2
```

`train_xla.py` is the XLA port of `train.py` (same model/math, different
harness): one process per chip via `torch_xla.launch`, `DistributedSampler`
over the shards, `MpDeviceLoader` prefetching, all-reduce → global grad clip →
step, bf16 autocast, logging through `xm.add_step_closure` (no host syncs in
the hot path), and master-only `xm.save` checkpoints. Per-chip batch is
`cfg.train.batch_size` (32) → **global batch 256**. Kaggle sessions preempt:
checkpoints are written every `ckpt_every` steps, resume with `--resume`.
