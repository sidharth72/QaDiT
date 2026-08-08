# QaDiT: A Latent Diffusion Transformer for Audio Generation (~160M)

**QaDiT** is a pure-PyTorch text-to-audio model: a ~160M-parameter Diffusion
Transformer (DiT-B) that generates 10.24 s of 16 kHz audio from a natural-language
caption. Diffusion runs in a compressed mel latent space from a frozen AudioLDM
VAE; a frozen FLAN-T5-large encodes the prompt; a frozen HiFi-GAN vocoder turns
decoded mel spectrograms into waveforms.

This README is a replication guide: environment setup, dataset precompute,
single-GPU / multi-GPU training, checkpointing, and inference.

---

## What the model is

| Piece | Role | Trainable? |
|---|---|---|
| FLAN-T5-large | Caption → token embeddings for cross-attention | Frozen |
| AudioLDM VAE | Mel ↔ latent `[8, 256, 16]` | Frozen |
| **QaDiT (DiT-B)** | Denoise latents conditioned on text (~160M) | **Trained** |
| AST + REPA projector | Early-training representation alignment | Projector trained; AST frozen |
| HiFi-GAN vocoder | Mel → waveform at inference | Frozen |

**Architecture (trainable DiT)**

- Latent grid: `[8, 256, 16]` → 2×2 patchify → **1024 tokens**
- Width 768, depth 12, 12 heads (DiT-B class)
- AdaLN-Zero blocks, T5 cross-attention, 10% caption dropout for CFG
- Objective: **v-prediction** on a cosine schedule with logit-normal `t` sampling
- **REPA**: align mid-depth DiT features to AST features; weight decays to 0 over training

**Inference chain**

```text
caption → T5 → c
z_T ~ N(0,I) → DDIM + CFG (EMA DiT) → z0
z0 / latent_scale → VAE.decode → mel → HiFi-GAN → .wav
```

Prompts work best when they look like [AudioCaps](https://audiocaps.github.io/)
captions: short, concrete sound events (“A dog barks while birds chirp”).

---

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Shared shapes and hyperparameters (single source of truth) |
| `dit.py` | DiT backbone + REPA projector |
| `diffusion.py` | Schedule, v-pred targets, DDIM + CFG sampler |
| `precompute.py` | AudioCaps → sharded latents / T5 / AST cache |
| `dataset.py` | Dataset + distributed samplers over the cache |
| `train.py` | Single-process trainer (+ `--smoke`) |
| `train_ddp.py` | Multi-GPU DDP trainer (1…N GPUs via `torchrun`) |
| `sample.py` | Caption → `.wav` |
| `requirements.txt` | Dependencies |

Design notes (optional reading): [`../AudioDiffusionModel.md`](../AudioDiffusionModel.md),
[`../ArchitectureFlow.md`](../ArchitectureFlow.md).

---

## 1. Environment

```bash
git clone <your-fork-or-repo> QaDiT
cd QaDiT   # or audio_dit/
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

**PyTorch / CUDA tips**

- Prefer a CUDA build that matches the **host driver** (`nvidia-smi` → CUDA Version).
- RTX 5090 (Blackwell, `sm_120`) needs a recent build (in practice **torch ≥ 2.8** with CUDA 12.8+ wheels). Example:

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

- Do not install a CPU-only torch wheel on a GPU machine if a CUDA image already provides one (e.g. Kaggle).

Optional for logging / checkpoint backup:

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
```

---

## 2. Smoke tests (do these first)

### 2.1 Architecture smoke (1 process, no data downloads)

```bash
python train.py --smoke
```

Trains a tiny DiT for 20 steps on synthetic tensors, then runs DDIM+CFG.
Expect `[smoke] PASS`.

### 2.2 DDP harness smoke (multi-GPU or CPU fallback)

```bash
# Use the number of GPUs you have; 2 is the common case
torchrun --standalone --nproc_per_node=2 train_ddp.py --smoke
```

Builds a synthetic cache, trains briefly, validates, and checkpoints.
Expect `[ddp] finished all 20 steps`.

On a fresh paid pod, always run §2.2 before the real job.

> **Do not** pass real `--data` with `--smoke`. Smoke uses a tiny latent geometry
> (`channels=4`); real AudioCaps latents are 8-channel and will crash with a
> channel mismatch.

---

## 3. Dataset: precompute the AudioCaps cache

Training never runs the VAE / T5 / AST online. You precompute once per split.

### 3.1 What gets written

```text
cache/
  train/
    shard_00000.pt ...
    meta.json          # latent_scale, shard_sizes, num_samples, ...
  validation/
    shard_00000.pt ...
    meta.json
```

Each sample stores (fp16 on disk where applicable):

| Tensor | Shape | Source |
|---|---|---|
| `latent` | `[8, 256, 16]` | AudioLDM VAE |
| `text_emb` | `[64, 1024]` | FLAN-T5-large |
| `text_mask` | `[64]` | tokenizer attention mask |
| `repa` | `[1024, 768]` | AST features on the DiT grid |
| `caption` | string | original AudioCaps text |

Disk: ~2 MB/clip → on the order of **~90 GB** for the full ~45k train split.
For a first pass use `--max-samples 2000`.

### 3.2 Run precompute

Needs GPU recommended (VAE/T5/AST are heavy on CPU):

```bash
python precompute.py --out ./cache --split train
python precompute.py --out ./cache --split validation
```

Useful flags:

```bash
python precompute.py --out ./cache --split train \
  --max-samples 2000 \
  --batch-size 16 \
  --shard-size 256 \
  --device cuda
```

Source dataset: Hugging Face `OpenSound/AudioCaps` (`datasets` library).

### 3.3 Sanity-check the frozen audio stack (recommended)

Before long training, round-trip a few clips
`wav → mel → VAE encode → decode → vocoder → wav` and **listen**.
That path is the quality ceiling of everything downstream.

### 3.4 Staging data on rented GPUs

If training on a network volume (e.g. RunPod `/workspace`), copy the cache to
**local NVMe** when possible and point `--data` there; keep `--out` / HF uploads
on durable storage. Network-volume I/O often shows up as GPU util flapping 0↔95%.

---

## 4. Effective batch size (read this before launching)

Target global batch used in this project:

```text
effective_global_batch = batch_size × grad_accum_steps × num_gpus = 256
```

Keep that product at **256** when changing GPU count or microbatch, so the
learning-rate schedule stays meaningful.

| GPUs | Example microbatch / GPU | Grad accum | Global batch |
|---:|---:|---:|---:|
| 1 | 8 | 32 | 256 |
| 2 × T4 (16 GB) | 2 | 64 | 256 |
| 2 × 5090 (32 GB) | 8 | 16 | 256 |
| 2 × 5090 (more VRAM free) | 16 | 8 | 256 |
| 4 | 8 | 8 | 256 |
| 8 | 8 | 4 | 256 |

Raise microbatch and lower `grad_accum` when memory allows — fewer accumulation
passes usually improve samples/sec.

---

## 5. Train — 1 GPU

Single-process loop (`train.py`):

```bash
python train.py --data ./cache --out ./runs/qadit
```

Resume:

```bash
python train.py --data ./cache --out ./runs/qadit \
  --resume ./runs/qadit/ckpt_0002000.pt
```

Defaults live in `config.py` (`TrainConfig`): lr `1e-4`, warmup `500`,
`total_steps` `20000`, REPA weight `0.5` decaying over `15000` steps, etc.
Disable REPA for an A/B by setting `repa_weight = 0.0` in `config.py`.

You can also use the DDP script on one GPU:

```bash
torchrun --standalone --nproc_per_node=1 train_ddp.py \
  --data ./cache --out ./runs/qadit \
  --batch-size 8 --grad-accum-steps 32
```

---

## 6. Train — 2 GPUs (recommended production path)

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --data ./cache \
  --out ./runs/qadit \
  --batch-size 8 \
  --grad-accum-steps 16 \
  --workers 4 \
  --val-every 2000 \
  --total-steps 24000 \
  --repa-decay-steps 15000 \
  --max-hours 7.5 \
  --resume auto \
  --hf-repo-id YOUR_USER/qadit-checkpoints \
  --hf-token "$HF_TOKEN" \
  --wandb-api-key "$WANDB_API_KEY" \
  --wandb-run-name qadit_2gpu
```

### 6.1 Kaggle 2 × T4

```bash
# If NCCL hangs at init:
export NCCL_P2P_DISABLE=1

torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --data /kaggle/input/audiocaps-precomputed \
  --out /kaggle/working/runs/qadit \
  --batch-size 2 --grad-accum-steps 64 \
  --workers 2
```

### 6.2 RunPod 2 × RTX 5090

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --data /tmp/audiocaps-cache \          # prefer local disk copy
  --out /workspace/runs/qadit \
  --batch-size 8 --grad-accum-steps 16 \
  --workers 8 \
  --total-steps 24000 \
  --max-hours 7.5 \
  --resume auto \
  --hf-repo-id YOUR_USER/qadit-checkpoints \
  --hf-token "$HF_TOKEN" \
  --wandb-api-key "$WANDB_API_KEY"
```

Confirm CUDA works before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 7. Train — N GPUs

Same entrypoint; only change `nproc_per_node` and rebalance microbatch / accum
so global batch stays 256:

```bash
N=4
torchrun --standalone --nproc_per_node=$N train_ddp.py \
  --data ./cache --out ./runs/qadit \
  --batch-size 8 --grad-accum-steps $((256 / (8 * N))) \
  --resume auto
```

Multi-node (sketch):

```bash
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  train_ddp.py --data ./cache --out ./runs/qadit \
  --batch-size 8 --grad-accum-steps 4
```

Resume must use the **same** `world_size`, `batch_size`, `grad_accum_steps`,
AMP mode, and dataset signature as the checkpoint.

---

## 8. Unattended training features (`train_ddp.py`)

| Flag / behavior | Meaning |
|---|---|
| `--smoke` | Tiny synthetic end-to-end DDP test |
| `--max-hours H` | Stop cleanly after H wall-clock hours (not a forced full-duration run) |
| `SIGTERM` / `SIGINT` | Checkpoint + force-upload, then exit |
| `--resume auto` | Newest local ckpt; pull from HF if local is empty |
| `--upload-every 1` | Upload every checkpoint (default) |
| `--replace-remote` | Keep only newest `ckpt_*.pt` on the Hub |
| `--keep-local N` | Local retention; never drops last ckpt if upload failed |
| `--total-steps`, `--ckpt-every`, `--log-every`, `--warmup-steps`, `--repa-decay-steps` | Override `config.py` |

Stop condition is whichever comes first: **`total_steps`** or **`--max-hours`**.

Checkpoints include model, projector, EMA, optimizer, scheduler, GradScaler,
data cursor, per-rank RNG, best-val bookkeeping, and W&B run id.

**Validation note:** if validation crashes on a shard/metadata mismatch, you can
finish a paid run with `--val-every 0` and fix the validation cache later.
Judge quality by EMA samples, not only train curves. With REPA, `loss_total`
often **rises** as `λ` decays to 0 while `loss_diff` stays roughly flat — that is
expected.

---

## 9. Inference

```bash
python sample.py \
  --ckpt ./runs/qadit/ckpt_0023999.pt \
  --cache ./cache \
  --prompt "A dog barks while birds chirp in the distance" \
  --steps 50 \
  --cfg-scale 4.0 \
  --seed 0 \
  --out dog_birds.wav
```

`--cache` is only needed for `train/meta.json` → `latent_scale` (must match training).

**Sampling knobs**

| Flag | Typical range | Effect |
|---|---|---|
| `--steps` | 50–100 | More DDIM steps; diminishing returns past ~100 |
| `--cfg-scale` | 4–7 | Stronger text adherence; too high (>8–10) gets harsh |
| `--seed` | any int | Different draws |

Colab: write a wav, then play it:

```python
from IPython.display import Audio, display
display(Audio("dog_birds.wav"))
```

Download a Hub checkpoint:

```python
from huggingface_hub import hf_hub_download, list_repo_files

repo = "YOUR_USER/qadit-checkpoints"
files = [f for f in list_repo_files(repo, repo_type="model")
         if f.startswith("checkpoints/") and f.endswith(".pt")]
path = hf_hub_download(repo, filename=sorted(files)[-1], repo_type="model",
                       local_dir="./checkpoints")
print(path)
```

---

## 10. Suggested replication checklist

1. `pip install -r requirements.txt` (CUDA torch matching the machine)
2. `python train.py --smoke`
3. `torchrun --standalone --nproc_per_node=<N> train_ddp.py --smoke`
4. Precompute `train` + `validation` (or start with `--max-samples 2000`)
5. Launch training with **global batch 256**
6. Confirm first checkpoint upload / local save
7. Sample with EMA weights on AudioCaps-like prompts
8. Sweep `--cfg-scale` / `--steps` / seeds before changing the model

---

## 11. Hyperparameters (defaults in `config.py`)

| Setting | Default |
|---|---|
| DiT | hidden 768, depth 12, heads 12, patch 2, REPA tap layer 4 |
| Diffusion | 1000 train steps, cosine schedule, v-pred, logit-normal `t` |
| CFG dropout | `p_uncond = 0.1` |
| LR | `1e-4`, warmup 500, cosine over `total_steps` |
| REPA | weight 0.5 → 0 over 15k steps |
| EMA | 0.9999 |
| Sample defaults | 50 DDIM steps, CFG 4.0 |
| Clip length | 10.24 s @ 16 kHz |

A solid first full run on 2×5090 used about **24k** optimizer steps (~7 hours
at ~1 update/s with global batch 256). Longer training and/or more captioned
data (e.g. Clotho, WavCaps) improve coverage beyond AudioCaps-style prompts.

---

## License / attribution

Frozen components come from their upstream Hugging Face repos (AudioLDM,
FLAN-T5, AST, SpeechT5 HiFi-GAN). AudioCaps is loaded via `OpenSound/AudioCaps`.
Cite those works if you publish results. QaDiT’s DiT, diffusion math, and
training harness in this folder are the project-specific code.
