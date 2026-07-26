"""
Dataset over the shards written by precompute.py.

Each shard is a single .pt file holding a few hundred samples as stacked
tensors, so reads are large and sequential (friendly to spinning disks, cloud
buckets and, later, to a grain/tf.data port on TPU).  Shards are cached in
memory with a tiny LRU so random access across shard boundaries stays cheap.
"""

from bisect import bisect_right
from collections import OrderedDict
import math
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, Sampler

class PrecomputedAudioCaps(Dataset):
    """Yields dicts with keys: latent, text_emb, text_mask, repa, caption.

    `latent` is returned ALREADY SCALED to ~unit variance (multiplied by
    meta.json's latent_scale), so train.py can treat it as diffusion-ready.
    """

    def __init__(self, root: str, split: str = "train", cache_shards: int = 2,
                 latent_scale: float | None = None):
        if cache_shards < 1:
            raise ValueError("cache_shards must be at least 1")
        self.dir = Path(root) / split
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found - run precompute.py --split {split} first")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.storage_latent_scale = float(self.meta["latent_scale"])
        # Held-out data must receive the same transform learned from train.
        # Passing train's scale avoids leaking validation-set statistics.
        self.latent_scale = (self.storage_latent_scale if latent_scale is None
                             else float(latent_scale))
        self.shard_files = [self.dir / name for name in self.meta["shards"]]

        self.shard_sizes = self._read_shard_sizes()
        self.shard_offsets = [0]
        for size in self.shard_sizes:
            self.shard_offsets.append(self.shard_offsets[-1] + size)

        self.cache_shards = cache_shards
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def _read_shard_sizes(self) -> list[int]:
        """Read sizes from metadata, including legacy-cache compatibility."""
        sizes = self.meta.get("shard_sizes")
        if sizes is not None:
            sizes = [int(size) for size in sizes]
        else:
            # Old caches did not record per-shard sizes. precompute.py always
            # wrote full shards followed by one remainder, so num_samples and
            # the number of shard files are enough to recover them without
            # reading the entire cache once per XLA rank.
            total = int(self.meta["num_samples"])
            n_shards = len(self.shard_files)
            if n_shards == 0:
                sizes = []
            else:
                full = int(self.meta.get(
                    "shard_size", math.ceil(total / n_shards)))
                last = total - full * (n_shards - 1)
                if last <= 0 or last > full:
                    raise ValueError(
                        "legacy cache has ambiguous shard sizes; regenerate "
                        "meta.json with the current precompute.py")
                sizes = [full] * (n_shards - 1) + [last]

        if len(sizes) != len(self.shard_files):
            raise ValueError("meta.json shard_sizes does not match shards")
        if any(size <= 0 for size in sizes):
            raise ValueError("all precomputed shards must be non-empty")
        if sum(sizes) != int(self.meta["num_samples"]):
            raise ValueError("meta.json shard sizes do not sum to num_samples")
        return sizes

    def _shard(self, si: int) -> dict:
        if si not in self._cache:
            if len(self._cache) >= self.cache_shards:
                self._cache.popitem(last=False)
            self._cache[si] = torch.load(self.shard_files[si], map_location="cpu")
        else:
            self._cache.move_to_end(si)
        return self._cache[si]

    def __len__(self) -> int:
        return self.shard_offsets[-1]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        si = bisect_right(self.shard_offsets, idx) - 1
        oi = idx - self.shard_offsets[si]
        shard = self._shard(si)
        return {
            # fp16 on disk -> fp32 for the training math; scaled to unit var.
            "latent": shard["latent"][oi].float() * self.latent_scale,
            "text_emb": shard["text_emb"][oi].float(),
            "text_mask": shard["text_mask"][oi].long(),
            "repa": shard["repa"][oi].float(),
            "caption": shard["caption"][oi],
        }


class ShardDistributedSampler(Sampler[int]):
    """Equal-length rank shards while retaining sequential shard locality.

    Every rank derives the same shuffled shard/sample order, then takes a
    strided view. Consequently, one global batch reads from a small number of
    shard files instead of causing a cache miss for almost every sample.
    """

    def __init__(self, dataset: PrecomputedAudioCaps, num_replicas: int,
                 rank: int, samples_per_step: int, seed: int = 0):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        if samples_per_step <= 0:
            raise ValueError("samples_per_step must be positive")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.samples_per_step = samples_per_step
        self.seed = seed
        self.epoch = 0
        self.start_index = 0
        global_step_samples = num_replicas * samples_per_step
        self.total_size = (len(dataset) // global_step_samples) * global_step_samples
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_start_index(self, start_index: int):
        if not 0 <= start_index <= self.num_samples:
            raise ValueError("sampler start_index is outside this rank's shard")
        self.start_index = start_index

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(
            len(self.dataset.shard_sizes), generator=generator).tolist()
        indices: list[int] = []
        for shard_idx in shard_order:
            start = self.dataset.shard_offsets[shard_idx]
            size = self.dataset.shard_sizes[shard_idx]
            within = torch.randperm(size, generator=generator).tolist()
            indices.extend(start + offset for offset in within)
        indices = indices[:self.total_size]
        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices[self.start_index:])


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation data across ranks without padding or duplicates."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))


def collate(batch: list[dict]) -> dict:
    """Stack tensors; keep captions as a plain list of strings."""
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "repa": torch.stack([b["repa"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
