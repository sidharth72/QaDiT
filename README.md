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
| `train_ddp.py` | Production trainer (2 x RTX 5090 or 2 x T4): NCCL DDP, FP16 AMP, gradient accumulation, distributed validation, exact resume, wall-clock budget and HF checkpoint mirroring |
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

## 5. Train with two GPUs (PyTorch DDP + mixed precision)

`train_ddp.py` keeps the model and diffusion math identical to `train.py` and
changes only the execution harness. `torchrun` runs one process per GPU, NCCL
DDP averages gradients, and CUDA autocast + `GradScaler` provide FP16.

### 5.1 Prove the harness first (30 seconds, no data needed)

Always run this on a fresh machine before starting a paid job. It builds a
synthetic cache, trains a tiny model, validates, checkpoints, resumes and (if
you pass HF/W&B credentials) exercises those too:

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py --smoke
```

A clean `[ddp] finished all 20 steps` means NCCL, DDP, AMP, the samplers,
validation and checkpointing are all working on this machine.

### 5.2 RunPod, 2 x RTX 5090

Blackwell needs a CUDA 12.8+ PyTorch build (**torch >= 2.8**); the trainer
checks the installed build has `sm_120` kernels and exits immediately if not,
rather than failing deep into a paid run.

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --data /workspace/cache \
    --out /workspace/runs/dit_b2 \
    --batch-size 8 --grad-accum-steps 16 \
    --max-hours 7.5 --resume auto \
    --hf-repo-id you/audio-dit-checkpoints --hf-token "$HF_TOKEN" \
    --wandb-api-key "$WANDB_API_KEY"
```

A 32 GB 5090 fits a much larger microbatch than a T4. Keep
`batch_size × grad_accum_steps × num_gpus` at **256** so the effective global
batch (and therefore the LR schedule) is unchanged: `8 × 16 × 2 = 256`.

### 5.3 Kaggle, 2 x T4

Same script; the 16 GB cards need the smaller microbatch. If NCCL hangs at
startup on Kaggle, export `NCCL_P2P_DISABLE=1` first.

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --data /kaggle/input/audiocaps-precomputed \
    --out /kaggle/working/runs/dit_b2 \
    --batch-size 2 --grad-accum-steps 64
```

### 5.4 Unattended-run behaviour

`--max-hours` stops cleanly at a wall-clock budget and `SIGTERM`/`SIGINT` (pod
eviction, Ctrl-C) does the same: the run finishes the current optimizer step,
writes a checkpoint, force-uploads it, and exits 0. All ranks agree on the stop
via a collective, so no rank is ever left waiting.

`--resume auto` continues from the newest checkpoint in `--out`, downloading it
back from the Hugging Face repo first if local storage was wiped. Restarting a
recycled pod is therefore the same command as the original launch.

Checkpoints are written atomically every `ckpt_every` optimizer updates and
carry model/projector/EMA, optimizer/scheduler, AMP scaler, exact data cursor,
per-rank CPU/CUDA RNG state, best validation state and the W&B run ID. Resume
refuses to proceed if the config, batch size, accumulation, world size, FP16
mode or dataset signature differ from the checkpoint, so a resumed run is never
silently a different experiment.

Each upload replaces older `ckpt_*.pt` files in the repo in the same commit, so
remote storage stays flat instead of growing ~2.6 GB per save. `--keep-local`
bounds local copies and never deletes the last one unless an upload succeeded.
(HF keeps deleted blobs in git history; squash or recreate the repo if the
accumulated history becomes a quota problem.)

Every `--val-every` optimizer updates, validation covers the split exactly once
across both GPUs (no DistributedSampler padding), disables CFG dropout, and logs
globally sample-weighted raw/EMA diffusion losses plus REPA loss. Validation
latents deliberately use the training split's scale, so held-out statistics
never alter the model's input transform.

`--total-steps`, `--ckpt-every`, `--log-every`, `--warmup-steps` and
`--repa-decay-steps` override `config.py` from the command line. They are part
of the config signature, so use identical values when resuming.
