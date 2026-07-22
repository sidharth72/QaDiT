"""
Dataset over the shards written by precompute.py.

Each shard is a single .pt file holding a few hundred samples as stacked
tensors, so reads are large and sequential (friendly to spinning disks, cloud
buckets and, later, to a grain/tf.data port on TPU).  Shards are cached in
memory with a tiny LRU so random access across shard boundaries stays cheap.
"""

from pathlib import Path
import json
import torch
from torch.utils.data import Dataset

class PrecomputedAudioCaps(Dataset):
    """Yields dicts with keys: latent, text_emb, text_mask, repa, caption.

    `latent` is returned ALREADY SCALED to ~unit variance (multiplied by
    meta.json's latent_scale), so train.py can treat it as diffusion-ready.
    """

    def __init__(self, root: str, split: str = "train", cache_shards: int = 2):
        self.dir = Path(root) / split
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found - run precompute.py --split {split} first")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.latent_scale = float(self.meta["latent_scale"])
        self.shard_files = [self.dir / name for name in self.meta["shards"]]

        # Index: global sample idx -> (shard idx, offset inside shard).
        # Shard sizes can vary (the last one is usually short), so read the
        # true length of each shard once up front.
        self.index: list[tuple[int, int]] = []
        for si, path in enumerate(self.shard_files):
            n = torch.load(path, map_location="cpu")["latent"].shape[0]
            self.index.extend((si, oi) for oi in range(n))

        self.cache_shards = cache_shards
        self._cache: dict[int, dict] = {}    # shard idx -> loaded dict (LRU)

    def _shard(self, si: int) -> dict:
        if si not in self._cache:
            if len(self._cache) >= self.cache_shards:
                self._cache.pop(next(iter(self._cache)))   # evict oldest
            self._cache[si] = torch.load(self.shard_files[si], map_location="cpu")
        return self._cache[si]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        si, oi = self.index[idx]
        shard = self._shard(si)
        return {
            # fp16 on disk -> fp32 for the training math; scaled to unit var.
            "latent": shard["latent"][oi].float() * self.latent_scale,
            "text_emb": shard["text_emb"][oi].float(),
            "text_mask": shard["text_mask"][oi].long(),
            "repa": shard["repa"][oi].float(),
            "caption": shard["caption"][oi],
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensors; keep captions as a plain list of strings."""
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "repa": torch.stack([b["repa"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
