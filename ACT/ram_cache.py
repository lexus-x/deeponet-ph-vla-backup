"""
ram_cache.py
============
Decode each unique dataset item once into RAM, then serve every later epoch from
memory instead of re-decoding parquet.

Why this is safe (no effect on training):
  A 30K-step / batch-64 run draws 1.92M samples from only ~53K unique frames, so
  every frame is decoded ~36x per run. The wrapped LeRobotDataset has
  image_transforms=None and is deterministic (verified ds[i]==ds[i]), so memoising
  each item's FULL output is byte-identical to the original — same frames, same
  action chunks, same order, same batches. Only the redundant re-decoding is
  removed. Steps, batch size, samples seen, and RNG are all unchanged.

The cache is built in the main process before the training DataLoader forks its
workers, so workers inherit it copy-on-write (shared, not duplicated).
"""
from __future__ import annotations
import time
import torch
from torch.utils.data import Dataset, DataLoader

HEADROOM_BYTES = 60 * 1024 ** 3  # keep free for model, prefetch buffers, OS


def _mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


class _Indexed(Dataset):
    """Yields (index, item) so the parallel prefill knows where each result goes."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return i, self.base[i]


def _contig(item):
    # Standalone contiguous tensors (not views into a shared buffer) so serve-time
    # collation / IPC sends only the small per-item tensor.
    return {k: (v.clone().contiguous() if torch.is_tensor(v) else v) for k, v in item.items()}


class RamCachedDataset(Dataset):
    def __init__(self, base):
        self._base = base
        self.cache = [None] * len(base)

    # delegate num_frames / stats / etc. to the wrapped dataset
    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        c = self.cache[i]
        return c if c is not None else self._base[i]  # fallback never hit after prefill

    def prefill(self, num_workers, log=print, batch_size=16):
        t0 = time.perf_counter()
        loader = DataLoader(_Indexed(self._base), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            prefetch_factor=(4 if num_workers > 0 else None),
                            persistent_workers=False, collate_fn=lambda b: b)
        n = len(self._base); done = 0; next = 10000
        for batch in loader:
            for idx, item in batch:
                self.cache[idx] = _contig(item)
                done += 1
            if done >= next:
                log(f"[ramcache] {done}/{n}  ({time.perf_counter()-t0:.0f}s)")
                next += 10000
        miss = sum(c is None for c in self.cache)
        assert miss == 0, f"prefill incomplete: {miss} items missing"
        log(f"[ramcache] cached {n} items in {(time.perf_counter()-t0)/60:.1f} min")
        return self


def maybe_ram_cache(ds, num_workers, log=print):
    """Wrap ds in a RAM cache if it comfortably fits; otherwise return ds unchanged.
    Returns (dataset, serve_workers). Serve needs few workers (no decode); prefill
    uses the full count."""
    probe = ds[0]
    per_item = sum(v.numel() * v.element_size() for v in probe.values() if torch.is_tensor(v))
    est = per_item * len(ds)
    avail = _mem_available_bytes()
    avail_s = f"{avail/1e9:.0f}" if avail else "?"
    log(f"[ramcache] est {est/1e9:.1f} GB for {len(ds)} items (avail {avail_s} GB)")
    if avail is None or est + HEADROOM_BYTES > avail:
        log("[ramcache] too large for RAM -> keeping on-the-fly decode")
        return ds, num_workers
    cached = RamCachedDataset(ds).prefill(num_workers=num_workers, log=log)
    return cached, min(8, num_workers)
