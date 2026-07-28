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
| `train.py` | Single-process GPU/CPU loop with a CPU-friendly `--smoke` mode |
| `train_ddp.py` | Production trainer for Kaggle 2 x T4: NCCL DDP, FP16 AMP, gradient accumulation, distributed validation and exact resume |
| `sample.py` | Inference: caption → T5 → DDIM+CFG → VAE decode → HiFi-GAN → `.wav` |

## Install

```bash
pip install -r requirements.txt
```

For the smoke test alone, `torch` is enough. Kaggle GPU images already include
a CUDA-enabled PyTorch build; do not replace it with a CPU-only wheel.

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

## 5. Train on Kaggle 2 x T4 (PyTorch DDP + mixed precision)

Upload the source files and precomputed cache as Kaggle Datasets, select the
**GPU T4 x2** accelerator, copy the source to a writable working directory,
and launch one process per GPU with `torchrun`. The cache must contain both
`train/` and `validation/`:

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --data /kaggle/input/audiocaps-precomputed \
    --out /kaggle/working/runs/dit_b2 \
    --batch-size 2 --grad-accum-steps 64 \
    --val-every 2000 --val-batch-size 2
```

`train_ddp.py` keeps the model and diffusion math unchanged. `torchrun` assigns
one process to each T4, NCCL DDP averages gradients, and CUDA autocast +
`GradScaler` use FP16 safely. The defaults use a microbatch of 2 per GPU and 64
accumulation passes, preserving an **effective global batch of 2 × 64 × 2 =
256** while fitting each T4's 16 GB VRAM. If memory allows, increase
`--batch-size` and reduce `--grad-accum-steps` by the same factor. If a run is
unstable, `--no-amp` is available for diagnosis but uses substantially more
memory.

Every `--val-every` optimizer updates, validation covers the split exactly once
across both GPUs (no DistributedSampler padding), disables CFG dropout, and
logs globally sample-weighted raw/EMA diffusion losses plus REPA loss. Validation
latents deliberately use the training split's scale; held-out statistics do not
alter the model's input transform.

Checkpoints are written atomically every `ckpt_every` optimizer updates and
include model/projector/EMA, optimizer/scheduler, exact data cursor, per-rank
CPU/CUDA RNG state, AMP scaler, best validation state, and W&B run ID. Resume
with
`--resume /path/to/ckpt_XXXXXXX.pt`; the batch/accumulation/world-size settings
and FP16 mode must match. The trainer also validates both cache metadata files
and sampler geometry before restoring an exact data cursor. Legacy checkpoints
still load, but their old sampler position or cache identity can only be
approximated.
