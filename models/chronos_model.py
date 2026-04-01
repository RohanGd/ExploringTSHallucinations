"""
Chronos wrapper for batch forecasting with optional activation saving.

Install: pip install chronos-forecasting
Model cards: https://huggingface.co/amazon/chronos-t5-{tiny,mini,small,base,large}

Chronos API
-----------
ChronosPipeline.predict(
    context           = List[Tensor] | Tensor,   # 1-D tensors, variable length OK
    prediction_length = int,
    num_samples       = int,
) -> Tensor  shape (B, num_samples, prediction_length)

Activation saving
-----------------
One HDF5 file is written per predict() call (i.e. per batch):
    activations/<save_activations>/<call_count>.h5

File structure:
    step_0/
        encoder.block.0.layer.0.SelfAttention.q   (n_positions, d_model)
        encoder.block.0.layer.0.SelfAttention.k
        ...
    step_1/
        ...

"step" corresponds to a decoder autoregressive step.
The encoder runs once (step_0), then the decoder runs forecast_horizon times.
"""

from typing import Literal
import os

import h5py
import numpy as np
import torch

ModelSize = Literal["tiny", "mini", "small", "base", "large"]


class ChronosModel:
    """
    Parameters
    ----------
    size             : Chronos model size (default "small").
    device           : "cuda", "cpu", or "mps".
    num_samples      : Monte-Carlo samples used to form the predictive median.
    dtype            : bfloat16 saves VRAM on modern GPUs; use float32 on CPU.
    save_activations : subdirectory name under activations/ for HDF5 files.
                       Set to None to disable activation saving entirely.
    """

    _HF_PREFIX = "amazon/chronos-t5"

    def __init__(
        self,
        size:             ModelSize = "small",
        device:           str = "cuda",
        num_samples:      int = 20,
        dtype:            torch.dtype = torch.bfloat16,
        save_activations: str | None = "latest",
    ):
        from chronos import ChronosPipeline          # lazy import

        model_id = f"{self._HF_PREFIX}-{size}"
        print(f"Loading Chronos ({model_id}) ...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map  = device,
            torch_dtype = dtype,
        )

        self.num_samples   = num_samples
        self.call_count    = 0
        self._current_step = 0          # tracks decoder autoregressive step
        self._h5_file      = None       # type: h5py.File | None

        # Activation hooks (optional)
        if save_activations is not None:
            self.activations_save_dir = os.path.join("activations", save_activations)
            os.makedirs(self.activations_save_dir, exist_ok=True)
            self._register_hooks()
        else:
            self.activations_save_dir = None

    # ------------------------------------------------------------------
    def _register_hooks(self):
        model = self.pipeline.model
        model.eval()

        # Forward hook on every named module
        for name, module in model.named_modules():
            module.register_forward_hook(self._make_activation_hook(name))

        # Hook on last decoder block to increment step counter
        last_decoder_block = model.model.decoder.block[-1]
        last_decoder_block.register_forward_hook(self._decoder_step_hook)

    def _decoder_step_hook(self, module, input, output):
        """Increment step counter after each decoder block forward pass."""
        self._current_step += 1

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            if self._h5_file is None:
                return

            # Unwrap tuple outputs (attention layers return (tensor, weights, ...))
            if isinstance(output, tuple):
                output = output[0]

            if not torch.is_tensor(output):
                return

            # FIX 1: cast bfloat16 -> float32 before .numpy()
            # numpy has no bfloat16 dtype; float32 preserves all information
            out = output.detach().cpu().to(torch.float32).numpy()

            group_path = f"step_{self._current_step}/{name}"
            try:
                self._h5_file.create_dataset(
                    group_path,
                    data        = out,
                    compression = "gzip",
                )
            except ValueError:
                # Dataset already exists: module called multiple times in one
                # step (e.g. shared embeddings). Silently skip duplicates.
                pass

        return hook

    # ------------------------------------------------------------------
    def predict(
        self,
        context:          torch.Tensor,
        forecast_horizon: int,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        context          : FloatTensor (B, L) -- NaN left-padded.
        forecast_horizon : int

        Returns
        -------
        np.ndarray (B, forecast_horizon) -- predictive median.
        """
        # Strip NaN padding: give Chronos each row's valid suffix
        context_list = [
            context[i][~torch.isnan(context[i])]
            for i in range(context.shape[0])
        ]

        # FIX 2: open h5py.File (not built-in open())
        # FIX 3: reset step counter per predict() call so each batch
        #        has consistent step_0=encoder, step_1..N=decoder steps
        if self.activations_save_dir is not None:
            h5_path = os.path.join(
                self.activations_save_dir, f"{self.call_count}.h5"
            )
            self._h5_file      = h5py.File(h5_path, "w")
            self._current_step = 0

        # forecast shape: (B, num_samples, forecast_horizon)
        # NOTE: context must be passed positionally, not as keyword arg
        forecast = self.pipeline.predict(
            context_list,
            prediction_length = forecast_horizon,
            num_samples       = self.num_samples,
        )

        # Close HDF5 and advance counter
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

        self.call_count += 1

        # Median over sample dimension -> (B, H)
        median = np.quantile(forecast.numpy(), 0.5, axis=1)
        return median.astype(np.float32)