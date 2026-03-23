"""
TimesFM wrapper for batch forecasting.

Install: pip install timesfm
Model card: https://huggingface.co/google/timesfm-1.0-200m

TimesFM API (v1.x)
------------------
tfm.forecast(
    inputs  = List[List[float]],   # variable-length context per series
    freq    = List[int],           # 0=high-freq, 1=medium, 2=low
) -> (point_forecast, quantile_forecast)
   point_forecast shape: (B, horizon_len)
"""

from typing import List

import numpy as np
import torch


# Frequency mapping: M4 group → TimesFM freq code
_GROUP_TO_FREQ = {
    "Hourly":     0,
    "Daily":      0,
    "Weekly":     1,
    "Monthly":    1,
    "Quarterly":  2,
    "Yearly":     2,
}


class TimesFMModel:
    """
    Parameters
    ----------
    model_id     : HuggingFace repo id.
    backend      : "gpu" | "cpu" | "tpu".
    horizon_len  : maximum horizon the model is initialised for.
                   Must be >= forecast_horizon used at inference time.
    batch_size   : per-core batch size passed to TimesFM.
    default_freq : freq code used when metadata group is unknown (0 = high).
    """

    def __init__(
        self,
        model_id:     str = "google/timesfm-1.0-200m",
        backend:      str = "gpu",
        horizon_len:  int = 64,
        batch_size:   int = 32,
        default_freq: int = 0,
    ):
        import timesfm                               # lazy import

        print(f"Loading TimesFM ({model_id}) …")
        self.tfm = timesfm.TimesFm(
            hparams = timesfm.TimesFmHparams(
                backend            = backend,
                per_core_batch_size = batch_size,
                horizon_len        = horizon_len,
            ),
            checkpoint = timesfm.TimesFmCheckpoint(
                huggingface_repo_id = model_id,
            ),
        )
        self.horizon_len  = horizon_len
        self.default_freq = default_freq

    # ------------------------------------------------------------------
    def predict(
        self,
        context:          torch.Tensor,
        forecast_horizon: int,
        metadata:         list | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        context          : FloatTensor (B, L) — NaN left-padded.
        forecast_horizon : int  (must be <= self.horizon_len)
        metadata         : optional list of dicts with 'group' key for
                           M4-aware frequency selection.

        Returns
        -------
        np.ndarray (B, forecast_horizon) — point forecast.
        """
        assert forecast_horizon <= self.horizon_len, (
            f"forecast_horizon ({forecast_horizon}) > horizon_len "
            f"({self.horizon_len}).  Re-initialise TimesFMModel with a "
            f"larger horizon_len."
        )

        # Strip NaN padding
        ctx_np = context.numpy()
        inputs: List[List[float]] = []
        for i in range(ctx_np.shape[0]):
            row  = ctx_np[i]
            mask = ~np.isnan(row)
            inputs.append(row[mask].tolist())

        # Frequency list
        if metadata is not None:
            freq = [
                _GROUP_TO_FREQ.get(m.get("group", ""), self.default_freq)
                for m in metadata
            ]
        else:
            freq = [self.default_freq] * len(inputs)

        point_forecast, _ = self.tfm.forecast(inputs, freq=freq)
        # point_forecast: (B, horizon_len) — trim to requested horizon
        return point_forecast[:, :forecast_horizon].astype(np.float32)