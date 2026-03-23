"""
evaluation/trend.py
-------------------
Trend-based hallucination metrics.

Core idea
---------
A model "halluccinates" the trend when it produces a forecast whose
overall direction (slope sign) disagrees with the ground-truth direction.

Three quantities are computed per series:

  context_slope   — slope of a linear fit to the context window.
  target_slope    — slope of a linear fit to the ground-truth target.
  pred_slope      — slope of a linear fit to the forecast.

From these we derive:

  TrendDirectionError (TDE)
      Binary: 1 if sign(pred_slope) ≠ sign(target_slope), else 0.
      A high aggregate TDE rate = the model frequently hallucinates
      the trend direction.

  TrendMagnitudeRatio (TMR)
      |pred_slope| / (|target_slope| + eps).
      TMR >> 1  → model over-extrapolates trend.
      TMR << 1  → model under-extrapolates (dampens trend).
      TMR ≈ 1   → model matches magnitude well.

  ContextTargetTrendAlignment (CTTA)
      Whether the *true* trend reverses between context and target.
      Useful as a conditioning variable — TDE is more forgivable if
      even the context slope pointed the wrong way.
"""

import numpy as np


def _linslope(arr: np.ndarray) -> np.ndarray:
    """
    Fit a line to each row and return the slope.

    Parameters
    ----------
    arr : (N, L) — NaN values are ignored (masked regression).

    Returns
    -------
    slopes : (N,)
    """
    N, L = arr.shape
    t    = np.arange(L, dtype=np.float64)
    slopes = np.full(N, np.nan)

    for i in range(N):
        row  = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        ti, yi = t[mask], row[mask].astype(np.float64)
        # Least-squares slope: cov(t,y)/var(t)
        tm, ym = ti.mean(), yi.mean()
        denom  = ((ti - tm) ** 2).sum()
        if denom < 1e-12:
            continue
        slopes[i] = ((ti - tm) * (yi - ym)).sum() / denom

    return slopes


def trend_metrics(
    context: np.ndarray,
    preds:   np.ndarray,
    targets: np.ndarray,
    eps:     float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    context : (N, L)  context window (may be NaN left-padded).
    preds   : (N, H)  model forecasts.
    targets : (N, H)  ground-truth targets.

    Returns
    -------
    {
      "context_slope" : (N,)  slope of context
      "target_slope"  : (N,)  slope of ground-truth target
      "pred_slope"    : (N,)  slope of forecast
      "TDE"           : (N,)  Trend Direction Error  ∈ {0, 1}
      "TMR"           : (N,)  Trend Magnitude Ratio
      "CTTA"          : (N,)  1 if context and target slopes share sign
    }
    """
    ctx_slope  = _linslope(context)
    tgt_slope  = _linslope(targets)
    pred_slope = _linslope(preds)

    # Trend Direction Error: did the model get the sign wrong?
    TDE = (np.sign(pred_slope) != np.sign(tgt_slope)).astype(np.float32)
    # NaN out cases where target is essentially flat (slope ≈ 0)
    flat = np.abs(tgt_slope) < eps
    TDE  = np.where(flat, np.nan, TDE)

    # Trend Magnitude Ratio
    TMR = np.abs(pred_slope) / (np.abs(tgt_slope) + eps)

    # Context-Target Trend Alignment: do context and target agree in direction?
    CTTA = (np.sign(ctx_slope) == np.sign(tgt_slope)).astype(np.float32)
    flat_ctx = np.abs(ctx_slope) < eps
    CTTA = np.where(flat_ctx | flat, np.nan, CTTA)

    return {
        "context_slope": ctx_slope.astype(np.float32),
        "target_slope" : tgt_slope.astype(np.float32),
        "pred_slope"   : pred_slope.astype(np.float32),
        "TDE"          : TDE.astype(np.float32),
        "TMR"          : TMR.astype(np.float32),
        "CTTA"         : CTTA.astype(np.float32),
    }