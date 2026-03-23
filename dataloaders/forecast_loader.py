"""
DataLoader factory for time-series forecasting datasets.

Collation strategy
------------------
Context tensors are **left-padded** with NaN to the longest sequence in
the mini-batch.  Both Chronos and TimesFM strip leading NaN values
before processing, so this is safe.

Targets are stacked without padding (all targets share the same
fixed forecast_horizon by construction).
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.base import TimeSeriesSample


# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[TimeSeriesSample]) -> dict:
    """
    Returns
    -------
    {
      "context"   : FloatTensor  (B, max_context_len)  — NaN left-padded
      "target"    : FloatTensor  (B, forecast_horizon)
      "item_ids"  : List[str]
      "metadata"  : List[dict]
    }
    """
    contexts = [torch.tensor(s.context, dtype=torch.float32) for s in batch]
    targets  = torch.tensor(
        np.stack([s.target for s in batch]), dtype=torch.float32
    )
    item_ids = [s.item_id  for s in batch]
    metadata = [s.metadata for s in batch]

    # Left-pad with NaN so all contexts reach the same length
    max_len = max(c.shape[0] for c in contexts)
    padded  = torch.full((len(batch), max_len), float("nan"))
    for i, c in enumerate(contexts):
        padded[i, -len(c):] = c          # right-align (= left-pad)

    return {
        "context":  padded,   # (B, max_context_len)
        "target":   targets,  # (B, forecast_horizon)
        "item_ids": item_ids,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
def get_dataloader(
    dataset:     Dataset,
    batch_size:  int = 32,
    shuffle:     bool = False,
    num_workers: int = 0,
    pin_memory:  bool = False,
) -> DataLoader:
    """
    Wrap any dataset that yields TimeSeriesSample objects in a DataLoader.

    Parameters
    ----------
    dataset     : a Dataset returning TimeSeriesSample items.
    batch_size  : mini-batch size.
    shuffle     : whether to shuffle (usually False for eval).
    num_workers : worker processes for parallel loading.
    pin_memory  : pin to CUDA-pinned memory (set True when using GPU).
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )