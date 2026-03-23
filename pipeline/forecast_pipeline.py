"""
ForecastPipeline
----------------
Iterates over a DataLoader, calls model.predict() per batch, and
writes CSV files per (model, dataset) pair:

  outputs/<model>_<dataset>_predictions.csv   — shape (N, forecast_horizon)
  outputs/<model>_<dataset>_targets.csv       — shape (N, forecast_horizon)
  outputs/<model>_<dataset>_context.csv       — shape (N, context_len), NaN-left-padded
  outputs/<model>_<dataset>_metadata.csv      — one metadata column per key

Row index = item_id.
Forecast columns = h1, h2, …, h{H}.
Context columns  = c1, c2, …, c{L}  (left-padded; c1 is the oldest/leftmost).
Metadata (waveform, group, etc.) is stored in a separate sidecar CSV.
"""

import os
import time
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
class Predictor(Protocol):
    """Duck-type interface: any model must implement predict()."""

    def predict(self, context: torch.Tensor, forecast_horizon: int,
                **kwargs) -> np.ndarray: ...


# ─────────────────────────────────────────────────────────────────────────────
class ForecastPipeline:
    """
    Parameters
    ----------
    model        : an object implementing Predictor (ChronosModel / TimesFMModel).
    model_name   : short string used in output filenames.
    forecast_horizon : forecast horizon (used for column names).
    output_dir   : directory where CSVs are written.
    """

    def __init__(
        self,
        model:            Predictor,
        model_name:       str,
        forecast_horizon: int = 64,
        output_dir:       str = "outputs",
    ):
        self.model            = model
        self.model_name       = model_name
        self.forecast_horizon = forecast_horizon
        self.output_dir       = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self, dataloader, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference over the full dataloader.

        Returns
        -------
        predictions : np.ndarray (N, forecast_horizon)
        targets     : np.ndarray (N, forecast_horizon)
        """
        all_preds:    list[np.ndarray] = []
        all_targets:  list[np.ndarray] = []
        all_contexts: list[np.ndarray] = []
        all_ids:      list[str]        = []
        all_metadata: list[dict]       = []

        t0 = time.time()

        for batch in tqdm(dataloader,
                          desc=f"{self.model_name} | {dataset_name}",
                          unit="batch"):
            context:  torch.Tensor = batch["context"]    # (B, L)
            targets:  np.ndarray   = batch["target"].numpy()
            item_ids: list[str]    = batch["item_ids"]
            metadata: list[dict]   = batch["metadata"]

            # ── forward pass ────────────────────────────────────────
            # Pass metadata so TimesFM can pick the right frequency
            try:
                preds = self.model.predict(
                    context          = context,
                    forecast_horizon = self.forecast_horizon,
                    metadata         = metadata,
                )
            except TypeError:
                # Model doesn't accept metadata kwarg (e.g. Chronos)
                preds = self.model.predict(
                    context          = context,
                    forecast_horizon = self.forecast_horizon,
                )

            all_preds.append(preds)
            all_targets.append(targets)
            all_contexts.append(context.numpy())   # (B, L_batch) — L may vary
            all_ids.extend(item_ids)
            all_metadata.extend(metadata)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  |  "
              f"{len(all_ids)} series  |  "
              f"{len(all_ids)/elapsed:.1f} series/s")

        predictions = np.concatenate(all_preds,   axis=0)  # (N, H)
        targets_arr = np.concatenate(all_targets, axis=0)  # (N, H)

        # Batches may have different context widths (NaN-padded to longest
        # series *within* each batch).  Re-pad all batches to the global max
        # before concatenating so every row has the same length.
        max_ctx_len = max(c.shape[1] for c in all_contexts)
        repadded = []
        for ctx_batch in all_contexts:
            pad_width = max_ctx_len - ctx_batch.shape[1]
            if pad_width > 0:
                pad = np.full((ctx_batch.shape[0], pad_width), np.nan,
                              dtype=np.float32)
                ctx_batch = np.concatenate([pad, ctx_batch], axis=1)
            repadded.append(ctx_batch)
        context_arr = np.concatenate(repadded, axis=0)      # (N, max_ctx_len)

        self._save(predictions, targets_arr, context_arr,
                   all_ids, all_metadata, dataset_name)
        return predictions, targets_arr

    # ------------------------------------------------------------------
    def _save(
        self,
        preds:    np.ndarray,
        targets:  np.ndarray,
        context:  np.ndarray,
        ids:      list[str],
        metadata: list[dict],
        dataset_name: str,
    ) -> None:
        prefix = f"{self.output_dir}/{self.model_name}_{dataset_name}"
        h_cols = [f"h{i+1}" for i in range(preds.shape[1])]
        c_cols = [f"c{i+1}" for i in range(context.shape[1])]

        # predictions
        pd.DataFrame(preds, index=ids, columns=h_cols).to_csv(
            f"{prefix}_predictions.csv", index_label="item_id"
        )
        # ground-truth targets
        pd.DataFrame(targets, index=ids, columns=h_cols).to_csv(
            f"{prefix}_targets.csv", index_label="item_id"
        )
        # context (NaN left-padded; c1 = oldest, c{L} = most recent)
        pd.DataFrame(context, index=ids, columns=c_cols).to_csv(
            f"{prefix}_context.csv", index_label="item_id"
        )
        # metadata sidecar
        pd.DataFrame(metadata, index=ids).to_csv(
            f"{prefix}_metadata.csv", index_label="item_id"
        )

        print(f"  Saved → {prefix}_{{predictions,targets,context,metadata}}.csv")