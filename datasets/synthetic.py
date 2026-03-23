"""
Synthetic time series dataset.

x(t) = signal(t) + trend(t) + noise(t)

Grid:
  waveforms  : sine, square, sawtooth, triangle, pulse          (5)
  num_periods: 8, 10, 12, 14, 16, 18, 20                        (7)
  slopes     : -0.01, 0.0, 0.01                                  (3)
  noise_stds : 0.0, 0.1, 0.2, 0.3, 0.4                          (5)
  -----------------------------------------------------------------
  Total      : 5 x 7 x 3 x 5 = 525
"""

from itertools import product
import numpy as np
from scipy import signal as sp_signal
from torch.utils.data import Dataset

from .base import TimeSeriesSample


WAVEFORMS   = ["sine", "square", "sawtooth", "triangle", "pulse"]
NUM_PERIODS = [8, 10, 12, 14, 16, 18, 20]
SLOPES      = [-0.01, 0.0, 0.01]
NOISE_STDS  = [0.0, 0.1, 0.2, 0.3, 0.4]


def _make_signal(waveform: str, num_periods: int, total_len: int,
                 context_len: int) -> np.ndarray:
    """
    Generate one period-normalised waveform over [0, total_len).

    Frequency is chosen so that exactly `num_periods` full cycles
    appear within the context window.
    """
    t = np.arange(total_len, dtype=np.float64)
    freq = num_periods / context_len          # cycles per sample
    phase = 2.0 * np.pi * freq * t

    if waveform == "sine":
        return np.sin(phase)
    elif waveform == "square":
        return sp_signal.square(phase)
    elif waveform == "sawtooth":
        return sp_signal.sawtooth(phase)
    elif waveform == "triangle":
        return sp_signal.sawtooth(phase, width=0.5)
    elif waveform == "pulse":
        # Narrow duty-cycle square wave (~10 %) as pulse proxy
        return sp_signal.square(phase, duty=0.1)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")


class SyntheticDataset(Dataset):
    """
    Deterministic 525-series synthetic dataset.

    Parameters
    ----------
    context_len      : int  — length of the context window fed to the model.
    forecast_horizon : int  — length of the target window.
    seed             : int  — RNG seed for reproducibility.
    """

    def __init__(
        self,
        context_len: int = 500,
        forecast_horizon: int = 64,
        seed: int = 42,
    ):
        self.context_len      = context_len
        self.forecast_horizon = forecast_horizon
        self.total_len        = context_len + forecast_horizon
        self._rng             = np.random.default_rng(seed)
        self.samples          = self._build()

        assert len(self.samples) == 525, \
            f"Expected 525 samples, got {len(self.samples)}"

    # ------------------------------------------------------------------
    def _build(self) -> list:
        samples = []
        t = np.arange(self.total_len, dtype=np.float64)

        combos = list(product(WAVEFORMS, NUM_PERIODS, SLOPES, NOISE_STDS))
        for idx, (waveform, num_periods, slope, noise_std) in enumerate(combos):

            sig   = _make_signal(waveform, num_periods,
                                 self.total_len, self.context_len)
            trend = slope * t
            noise = (self._rng.normal(0.0, noise_std, self.total_len)
                     if noise_std > 0 else np.zeros(self.total_len))

            x = (sig + trend + noise).astype(np.float32)

            samples.append(TimeSeriesSample(
                context  = x[: self.context_len],
                target   = x[self.context_len :],
                item_id  = f"synthetic_{idx:04d}",
                metadata = {
                    "waveform":    waveform,
                    "num_periods": num_periods,
                    "slope":       slope,
                    "noise_std":   noise_std,
                },
            ))
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TimeSeriesSample:
        return self.samples[idx]