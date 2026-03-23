"""
evaluation/accuracy.py
----------------------
Standard point-forecast accuracy metrics.

All functions operate on np.ndarray of shape (N, H) and return
per-series scores of shape (N,), so you can slice by metadata later
(e.g. accuracy per waveform type, per M4 group, etc.).

NaN handling
------------
Target values may contain NaN (M4 series shorter than forecast_horizon
are tail-padded with NaN).  Every metric ignores NaN positions.
"""

import numpy as np


def _mask(preds: np.ndarray, targets: np.ndarray):
    """Return boolean mask where neither pred nor target is NaN."""
    return ~(np.isnan(preds) | np.isnan(targets))


# ── element-wise absolute error ───────────────────────────────────────────────
def mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean Absolute Error, shape (N,)."""
    m = _mask(preds, targets)
    err = np.where(m, np.abs(preds - targets), np.nan)
    return np.nanmean(err, axis=1)


def rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error, shape (N,)."""
    m = _mask(preds, targets)
    sq = np.where(m, (preds - targets) ** 2, np.nan)
    return np.sqrt(np.nanmean(sq, axis=1))


# ── percentage errors ─────────────────────────────────────────────────────────
def mape(preds: np.ndarray, targets: np.ndarray,
         eps: float = 1e-8) -> np.ndarray:
    """
    Mean Absolute Percentage Error, shape (N,).
    Skips positions where |target| < eps to avoid division by zero.
    """
    m = _mask(preds, targets) & (np.abs(targets) >= eps)
    err = np.where(m, np.abs((preds - targets) / targets), np.nan)
    return np.nanmean(err, axis=1) * 100.0


def smape(preds: np.ndarray, targets: np.ndarray,
          eps: float = 1e-8) -> np.ndarray:
    """
    Symmetric MAPE, shape (N,).
    sMAPE = 200 * |p - t| / (|p| + |t| + eps)
    Bounded in [0, 200].
    """
    m = _mask(preds, targets)
    denom = np.abs(preds) + np.abs(targets) + eps
    err   = np.where(m, 200.0 * np.abs(preds - targets) / denom, np.nan)
    return np.nanmean(err, axis=1)


def compute_all_accuracy(
    preds:   np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convenience wrapper returning all metrics in a single dict.

    Returns
    -------
    {
      "mae"  : (N,)
      "rmse" : (N,)
      "mape" : (N,)
      "smape": (N,)
    }
    """
    return {
        "mae"  : mae  (preds, targets),
        "rmse" : rmse (preds, targets),
        "mape" : mape (preds, targets),
        "smape": smape(preds, targets),
    }