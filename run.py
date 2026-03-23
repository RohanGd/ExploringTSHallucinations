"""
run.py — main entry point for hallucination detection experiments.

Usage examples
--------------
# Full run (both models, both datasets)
python run.py

# Only synthetic dataset, only Chronos
python run.py --datasets synthetic --models chronos

# M4 monthly only, large batch on GPU
python run.py --datasets m4 --m4_groups Monthly --models chronos timesfm \
              --batch_size 24 --device cuda

# CPU-only smoke test
python run.py --datasets m4 --m4_groups Monthly --models chronos --batch_size 24 --device cuda
"""

import argparse
import torch

from datasets.synthetic     import SyntheticDataset
from datasets.m4            import M4Dataset, M4_GROUPS
from dataloaders.forecast_loader import get_dataloader
from models.chronos_model   import ChronosModel
from models.timesfm_model   import TimesFMModel
from pipeline.forecast_pipeline import ForecastPipeline


# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_LEN      = 500
FORECAST_HORIZON = 64


# ─────────────────────────────────────────────────────────────────────────────
def build_datasets(args) -> dict:
    ds = {}
    if "synthetic" in args.datasets:
        print("Building synthetic dataset …")
        ds["synthetic"] = SyntheticDataset(
            context_len      = CONTEXT_LEN,
            forecast_horizon = FORECAST_HORIZON,
        )
        print(f"  {len(ds['synthetic'])} series.")

    if "m4" in args.datasets:
        groups = args.m4_groups or M4_GROUPS
        ds["m4"] = M4Dataset(
            groups           = groups,
            context_len      = CONTEXT_LEN,
            forecast_horizon = FORECAST_HORIZON,
            data_dir         = args.m4_data_dir,
        )
    return ds


def build_models(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    if "chronos" in args.models:
        models["chronos"] = ChronosModel(
            size        = args.chronos_size,
            device      = device,
            num_samples = args.chronos_num_samples,
        )

    if "timesfm" in args.models:
        backend = "gpu" if "cuda" in device else "cpu"
        models["timesfm"] = TimesFMModel(
            model_id    = args.timesfm_model_id,
            backend     = backend,
            horizon_len = FORECAST_HORIZON,
            batch_size  = args.batch_size,
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    datasets = build_datasets(args)
    models   = build_models(args)

    for model_name, model in models.items():
        pipeline = ForecastPipeline(
            model            = model,
            model_name       = model_name,
            forecast_horizon = FORECAST_HORIZON,
            output_dir       = args.output_dir,
        )
        for dataset_name, dataset in datasets.items():
            loader = get_dataloader(
                dataset,
                batch_size  = args.batch_size,
                shuffle     = False,
                num_workers = args.num_workers,
                pin_memory  = "cuda" in (args.device or ""),
            )
            pipeline.run(loader, dataset_name=dataset_name)

    print("\nAll done. Results in:", args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS hallucination detection pipeline")

    # ── datasets ──────────────────────────────────────────────────────
    parser.add_argument(
        "--datasets", nargs="+", default=["synthetic", "m4"],
        choices=["synthetic", "m4"],
        help="Which datasets to evaluate on.",
    )
    parser.add_argument(
        "--m4_groups", nargs="+", default=None, choices=M4_GROUPS,
        help="M4 frequency groups to include (default: all).",
    )
    parser.add_argument(
        "--m4_data_dir", default="data/m4",
        help="Directory for datasetsforecast to cache M4 data.",
    )

    # ── models ────────────────────────────────────────────────────────
    parser.add_argument(
        "--models", nargs="+", default=["chronos", "timesfm"],
        choices=["chronos", "timesfm"],
    )
    parser.add_argument(
        "--chronos_size", default="small",
        choices=["tiny", "mini", "small", "base", "large"],
    )
    parser.add_argument(
        "--chronos_num_samples", type=int, default=20,
        help="Monte-Carlo samples for Chronos predictive median.",
    )
    parser.add_argument(
        "--timesfm_model_id", default="google/timesfm-1.0-200m",
    )

    # ── runtime ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device",      default=None,
                        help="cuda | cpu | mps  (auto-detected if not set).")
    parser.add_argument("--output_dir",  default="outputs")

    main(parser.parse_args())