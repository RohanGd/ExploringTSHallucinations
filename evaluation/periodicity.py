"""
evaluation/periodicity.py
--------------------------
Periodicity-based hallucination metrics.

Core idea
---------
We identify the *dominant frequency* (peak FFT bin) of both the ground-
truth target and the forecast, then measure how well they agree.

A model "hallucinates" periodicity when it invents or omits dominant
cycles that are clearly present in the context.

Metrics
-------

  DominantFreqError (DFE)
      |dominant_freq(pred) - dominant_freq(target)| / fs
      Normalised to [0, 0.5].  Zero = perfect match.

  FreqSpectrumCorr (FSC)
      Pearson correlation between the FFT magnitude spectra of
      pred and target (both trimmed to the positive-frequency half).
      FSC ∈ [-1, 1]; high FSC → the model reproduces the spectral shape.

  ContextFreqConsistency (CFC)
      Whether the dominant frequency of the *context* matches that of
      the *target*.  Used as a conditioning variable:
      if CFC is low the model has a harder task.

Notes
-----
* We apply a Hann window before FFT to reduce spectral leakage.
* For series shorter than 4 samples the metric returns NaN.
* For the synthetic dataset the ground-truth frequency is also available
  directly from metadata (num_periods / context_len); you can compare
  the FFT-recovered frequency against the analytical one as a sanity check.
"""

import numpy as np


def _dominant_freq(arr: np.ndarray) -> np.ndarray:
    """
    Dominant normalised frequency (cycles/sample) for each row.

    Parameters
    ----------
    arr : (N, L) — NaN values are linearly interpolated before FFT.

    Returns
    -------
    freqs : (N,)  dominant frequency in [0, 0.5]
    """
    N, L = arr.shape
    if L < 4:
        return np.full(N, np.nan)

    window = np.hanning(L)
    freqs  = np.fft.rfftfreq(L)          # shape (L//2 + 1,)
    result = np.full(N, np.nan)

    for i in range(N):
        row  = arr[i].astype(np.float64)
        mask = ~np.isnan(row)
        if mask.sum() < 4:
            continue
        # Linear interpolation over NaN gaps
        if not mask.all():
            idx  = np.arange(L)
            row  = np.interp(idx, idx[mask], row[mask])

        spec = np.abs(np.fft.rfft(row * window))
        # Exclude DC bin (index 0)
        spec[0] = 0.0
        result[i] = freqs[np.argmax(spec)]

    return result


def _magnitude_spectrum(arr: np.ndarray) -> np.ndarray:
    """
    Returns FFT magnitude spectra, shape (N, L//2+1).
    NaN rows → all-zero spectrum.
    """
    N, L = arr.shape
    window = np.hanning(L)
    out    = np.zeros((N, L // 2 + 1))

    for i in range(N):
        row  = arr[i].astype(np.float64)
        mask = ~np.isnan(row)
        if mask.sum() < 4:
            continue
        if not mask.all():
            idx = np.arange(L)
            row = np.interp(idx, idx[mask], row[mask])
        out[i] = np.abs(np.fft.rfft(row * window))

    return out


def _row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between corresponding rows. Returns (N,)."""
    N   = A.shape[0]
    out = np.full(N, np.nan)
    for i in range(N):
        a, b = A[i], B[i]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def periodicity_metrics(
    context: np.ndarray,
    preds:   np.ndarray,
    targets: np.ndarray,
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
      "dominant_freq_context" : (N,)
      "dominant_freq_target"  : (N,)
      "dominant_freq_pred"    : (N,)
      "DFE"                   : (N,)  Dominant Freq Error  ∈ [0, 0.5]
      "FSC"                   : (N,)  Freq Spectrum Corr   ∈ [-1, 1]
      "CFC"                   : (N,)  Context Freq Consistency ∈ [0, 0.5]
    }
    """
    df_ctx = _dominant_freq(context)
    df_tgt = _dominant_freq(targets)
    df_prd = _dominant_freq(preds)

    # Dominant Freq Error
    DFE = np.abs(df_prd - df_tgt)

    # Freq Spectrum Correlation
    spec_tgt = _magnitude_spectrum(targets)
    spec_prd = _magnitude_spectrum(preds)
    FSC      = _row_corr(spec_prd, spec_tgt)

    # Context Freq Consistency
    CFC = np.abs(df_ctx - df_tgt)

    return {
        "dominant_freq_context": df_ctx.astype(np.float32),
        "dominant_freq_target" : df_tgt.astype(np.float32),
        "dominant_freq_pred"   : df_prd.astype(np.float32),
        "DFE"                  : DFE.astype(np.float32),
        "FSC"                  : FSC.astype(np.float32),
        "CFC"                  : CFC.astype(np.float32),
    }