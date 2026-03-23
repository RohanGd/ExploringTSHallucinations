"""
evaluation/hallucination_score.py
----------------------------------
Composite Hallucination Score (CHS).

Motivation
----------
Individual metrics (TDE, DFE, IER, WD) each capture a different failure
mode.  CHS combines them into a single per-series scalar in [0, 1] so
series can be ranked by "how hallucinatory" the model was.

Scoring procedure
-----------------
Each component is first scaled to [0, 1]:

  TDE  ∈ {0,1}     → already in range              weight: w_trend
  DFE  ∈ [0, 0.5]  → divide by 0.5                 weight: w_period
  IER  ∈ [0, 1]    → already in range              weight: w_dist
  WD_norm          → min-max across the batch       weight: w_dist

Then:
  CHS_i = w_trend * TDE_i
        + w_period * DFE_norm_i
        + w_dist * (α * IER_i + (1-α) * WD_norm_i)

Default weights: w_trend=0.35, w_period=0.35, w_dist=0.30, α=0.5.

These are reasonable defaults — you should treat them as hyperparameters
and run ablations to see which component drives hallucination the most
for each model/dataset combination.

The module also exposes `hallucination_report()` which calls all metric
modules, computes CHS, and returns a single consolidated DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .accuracy     import compute_all_accuracy
from .trend        import trend_metrics
from .periodicity  import periodicity_metrics
from .distribution import distribution_metrics


# ─────────────────────────────────────────────────────────────────────────────
def _minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise ignoring NaN."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn + eps)


# ─────────────────────────────────────────────────────────────────────────────
def composite_hallucination_score(
    trend_dict: dict,
    period_dict: dict,
    dist_dict: dict,
    w_trend:  float = 0.35,
    w_period: float = 0.35,
    w_dist:   float = 0.30,
    alpha:    float = 0.50,   # IER vs WD mixing weight
) -> np.ndarray:
    """
    Compute CHS from pre-computed metric dicts.

    Returns
    -------
    CHS : np.ndarray (N,)  in [0, 1]
    """
    TDE     = np.nan_to_num(trend_dict ["TDE"],  nan=0.5)   # uncertain → 0.5
    DFE_n   = np.nan_to_num(period_dict["DFE"] / 0.5, nan=0.5).clip(0, 1)
    IER     = np.nan_to_num(dist_dict  ["IER"],  nan=0.0)
    WD_n    = _minmax(np.nan_to_num(dist_dict["WD_pred"], nan=0.0))

    dist_score = alpha * IER + (1.0 - alpha) * WD_n

    CHS = (w_trend * TDE + w_period * DFE_n + w_dist * dist_score)
    return CHS.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def hallucination_report(
    context:   np.ndarray,
    preds:     np.ndarray,
    targets:   np.ndarray,
    item_ids:  list[str],
    metadata:  list[dict] | None = None,
    chs_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Full evaluation pipeline for one (model, dataset) pair.

    Parameters
    ----------
    context   : (N, L)   context window (NaN-padded OK)
    preds     : (N, H)   model forecasts
    targets   : (N, H)   ground-truth targets
    item_ids  : list of N strings
    metadata  : optional list of N dicts (waveform, group, etc.)
    chs_kwargs: override default CHS weights

    Returns
    -------
    pd.DataFrame with columns:
      item_id, [metadata cols], mae, rmse, mape, smape,
      context_slope, target_slope, pred_slope, TDE, TMR, CTTA,
      dominant_freq_context, dominant_freq_target, dominant_freq_pred,
      DFE, FSC, CFC,
      WD_pred, WD_target, IER, MS,
      CHS
    """
    print("  Computing accuracy metrics …")
    acc = compute_all_accuracy(preds, targets)

    print("  Computing trend metrics …")
    trnd = trend_metrics(context, preds, targets)

    print("  Computing periodicity metrics …")
    per = periodicity_metrics(context, preds, targets)

    print("  Computing distribution metrics …")
    dist = distribution_metrics(context, preds, targets)

    print("  Computing composite hallucination score …")
    chs = composite_hallucination_score(
        trnd, per, dist, **(chs_kwargs or {})
    )

    # ── assemble DataFrame ────────────────────────────────────────────
    rows = {"item_id": item_ids}

    if metadata is not None:
        meta_df = pd.DataFrame(metadata, index=item_ids)
        for col in meta_df.columns:
            rows[col] = meta_df[col].values

    for d in [acc, trnd, per, dist]:
        rows.update(d)

    rows["CHS"] = chs

    df = pd.DataFrame(rows)
    df = df.set_index("item_id")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def summarise(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """
    Aggregate the report DataFrame.

    If group_by is given (e.g. ["waveform"] or ["group"]), compute per-group
    means; otherwise return global means.

    Returns a DataFrame of mean values.
    """
    metric_cols = [
        "mae", "rmse", "mape", "smape",
        "TDE", "TMR", "DFE", "FSC", "CFC",
        "WD_pred", "WD_target", "IER", "MS",
        "CHS",
    ]
    cols = [c for c in metric_cols if c in df.columns]

    if group_by:
        valid_groups = [g for g in group_by if g in df.columns]
        return df[valid_groups + cols].groupby(valid_groups).mean()
    else:
        return df[cols].mean().to_frame("mean").T