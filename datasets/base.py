from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimeSeriesSample:
    """
    Unified contract between datasets and models.

    context : np.ndarray of shape (L,) where L <= context_len.
              May be shorter than context_len for short M4 series.
              NaN padding is applied by the collate_fn, not here.
    target  : np.ndarray of shape (forecast_horizon,).
              May contain NaN at the tail if the M4 test horizon
              is shorter than forecast_horizon.
    item_id : unique string identifier for the time series.
    metadata: arbitrary dict for waveform type, M4 group, etc.
    """
    context:  np.ndarray
    target:   np.ndarray
    item_id:  str
    metadata: dict = field(default_factory=dict)