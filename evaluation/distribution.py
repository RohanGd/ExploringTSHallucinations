"""
evaluation/distribution.py
---------------------------
Distribution-based hallucination metrics.

Core idea
---------
A forecast that is statistically inconsistent with the observed context
is a candidate hallucination.  We compare the marginal distribution of
the forecast against the distribution of a trailing window of the context
(the "local context tail").

Metrics
-------

  WassersteinDistance (WD)
      1-Wasserstein (Earth Mover's) distance between the empirical
      distributions of the forecast and the context tail.
      Large WD → forecast lives in a very different value range.

  IQREscapeRate (IER)
      Fraction of forecast steps that fall outside the
      [Q1 - 1.5*IQR, Q3 + 1.5*IQR] fence of the context tail.
      IER = 0 → forecast stays in-distribution.
      IER > 0 → some steps escape the context's "normal" range.
      This is a soft outlier detector: unlike hard clipping it allows
      for trend-driven excursions and is robust to skewed distributions.

  MeanShift (MS)
      (mean(pred) - mean(ctx_tail)) / (std(ctx_tail) + eps)
      Signed; positive = upward shift, negative = downward.
      Captures systematic bias rather than spread mismatch.

Context tail length
-------------------
We use the last `tail_fraction` of the valid context as the reference
window.  Default is 0.2 (last 20%), i.e. ~100 steps for L=500.
Using the full context can wash out local level shifts.
"""

import numpy as np


def _tail(context: np.ndarray, tail_fraction: float = 0.2) -> np.ndarray:
    """
    Extract the last `tail_fraction` valid (non-NaN) points per row.

    Returns a list of 1-D arrays (variable length).
    """
    tails = []
    for row in context:
        valid = row[~np.isnan(row)]
        n     = max(4, int(len(valid) * tail_fraction))
        tails.append(valid[-n:])
    return tails


def _wasserstein1d(u: np.ndarray, v: np.ndarray) -> float:
    """1-Wasserstein distance between two 1-D empirical distributions."""
    us = np.sort(u)
    vs = np.sort(v)
    # Interpolate to equal length
    n  = max(len(us), len(vs))
    ui = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(us)), us)
    vi = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(vs)), vs)
    return float(np.mean(np.abs(ui - vi)))


def distribution_metrics(
    context:       np.ndarray,
    preds:         np.ndarray,
    targets:       np.ndarray,
    tail_fraction: float = 0.2,
    eps:           float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    context       : (N, L)  context window (NaN left-padded OK).
    preds         : (N, H)  model forecasts.
    targets       : (N, H)  ground-truth targets (used for target WD).
    tail_fraction : float   fraction of context used as reference tail.

    Returns
    -------
    {
      "WD_pred"   : (N,)  Wasserstein dist(forecast, ctx_tail)
      "WD_target" : (N,)  Wasserstein dist(target,   ctx_tail)  [oracle]
      "IER"       : (N,)  IQR Escape Rate of forecast w.r.t. ctx_tail
      "MS"        : (N,)  Mean Shift (normalised)
    }
    """
    N = context.shape[0]
    WD_pred   = np.full(N, np.nan)
    WD_target = np.full(N, np.nan)
    IER       = np.full(N, np.nan)
    MS        = np.full(N, np.nan)

    tails = _tail(context, tail_fraction)

    for i in range(N):
        tail = tails[i]
        if len(tail) < 4:
            continue

        pred_i = preds[i][~np.isnan(preds[i])]
        tgt_i  = targets[i][~np.isnan(targets[i])]
        if len(pred_i) < 1:
            continue

        # ── Wasserstein ───────────────────────────────────────────────
        WD_pred[i]   = _wasserstein1d(pred_i, tail)
        if len(tgt_i) >= 1:
            WD_target[i] = _wasserstein1d(tgt_i, tail)

        # ── IQR Escape Rate ──────────────────────────────────────────
        Q1, Q3 = np.percentile(tail, [25, 75])
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        IER[i] = np.mean((pred_i < lower) | (pred_i > upper))

        # ── Mean Shift ───────────────────────────────────────────────
        tail_std = tail.std()
        MS[i]    = (pred_i.mean() - tail.mean()) / (tail_std + eps)

    return {
        "WD_pred"  : WD_pred.astype(np.float32),
        "WD_target": WD_target.astype(np.float32),
        "IER"      : IER.astype(np.float32),
        "MS"       : MS.astype(np.float32),
    }