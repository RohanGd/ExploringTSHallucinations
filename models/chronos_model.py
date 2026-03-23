"""
Chronos wrapper for batch forecasting.

Install: pip install chronos-forecasting
Model cards: https://huggingface.co/amazon/chronos-t5-{tiny,mini,small,base,large}

Chronos API
-----------
ChronosPipeline.predict(
    context           = List[Tensor] | Tensor,   # 1-D tensors, variable length OK
    prediction_length = int,
    num_samples       = int,
) -> Tensor  shape (B, num_samples, prediction_length)
"""

from typing import Literal

import numpy as np
import torch

ModelSize = Literal["tiny", "mini", "small", "base", "large"]


class ChronosModel:
    """
    Parameters
    ----------
    size        : Chronos model size (default "small").
    device      : "cuda", "cpu", or "mps".
    num_samples : Monte-Carlo samples used to form the predictive median.
    dtype       : bfloat16 saves VRAM on modern GPUs; use float32 on CPU.
    """

    _HF_PREFIX = "amazon/chronos-t5"

    def __init__(
        self,
        size:        ModelSize = "small",
        device:      str = "cuda",
        num_samples: int = 20,
        dtype:       torch.dtype = torch.bfloat16,
    ):
        from chronos import ChronosPipeline          # lazy import

        model_id = f"{self._HF_PREFIX}-{size}"
        print(f"Loading Chronos ({model_id}) …")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map = device,
            dtype = dtype,
        )
        self.num_samples = num_samples

    # ------------------------------------------------------------------
    def predict(
        self,
        context: torch.Tensor,
        forecast_horizon: int,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        context          : FloatTensor (B, L) — NaN left-padded.
        forecast_horizon : int

        Returns
        -------
        np.ndarray (B, forecast_horizon) — predictive median.
        """
        # Strip NaN padding: give Chronos each row's valid suffix
        context_list = [
            context[i][~torch.isnan(context[i])]
            for i in range(context.shape[0])
        ]

        # forecast shape: (B, num_samples, forecast_horizon)
        # NOTE: context must be passed positionally — not as a keyword arg
        forecast = self.pipeline.predict(
            context_list,
            prediction_length = forecast_horizon,
            num_samples       = self.num_samples,
        )

        # Median over sample dimension
        median = np.quantile(forecast.numpy(), 0.5, axis=1)   # (B, H)
        return median.astype(np.float32)