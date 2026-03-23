"""
evaluate_from_csv.py
--------------------
Load pre-saved prediction / target CSVs from outputs/ and run the
full hallucination evaluation without re-running inference.

Usage
-----
# Evaluate all CSVs in outputs/
python evaluate_from_csv.py

# Evaluate a specific pair
python evaluate_from_csv.py --model chronos --dataset synthetic

# Custom output dir
python evaluate_from_csv.py --output_dir results/
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# Make sure the evaluation package is on the path
sys.path.insert(0, os.path.dirname(__file__))
from evaluation import hallucination_report, summarise


# ─────────────────────────────────────────────────────────────────────────────
def discover_pairs(output_dir: str) -> list[tuple[str, str]]:
    """
    Scan output_dir for *_predictions.csv files and return
    list of (model_name, dataset_name) tuples.
    """
    pattern = re.compile(r"^(.+?)_(.+?)_predictions\.csv$")
    pairs   = []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return sorted(set(pairs))


def load_pair(output_dir: str, model: str, dataset: str):
    """
    Load predictions, targets, and metadata CSVs.
    Returns (preds, targets, context, item_ids, metadata_list).
    Context is set to NaN (not saved) — distribution/trend metrics that
    need context will degrade gracefully; to preserve context save it in run.py.
    """
    prefix = f"{output_dir}/{model}_{dataset}"

    preds_df   = pd.read_csv(f"{prefix}_predictions.csv", index_col="item_id")
    targets_df = pd.read_csv(f"{prefix}_targets.csv",     index_col="item_id")

    meta_path = f"{prefix}_metadata.csv"
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path, index_col="item_id")
        metadata = meta_df.to_dict("records")
    else:
        metadata = [{}] * len(preds_df)

    item_ids = preds_df.index.tolist()
    preds    = preds_df.values.astype(np.float32)
    targets  = targets_df.values.astype(np.float32)


    # Load context if available, otherwise fall back to NaN placeholder
    ctx_path = f"{prefix}_context.csv"
    if os.path.exists(ctx_path):
        context = pd.read_csv(ctx_path, index_col="item_id").values.astype(np.float32)
    else:
        print(f"  Warning: no context CSV found at {ctx_path}. "
              "Trend/distribution metrics will be skipped.")
        context = np.full((len(preds), 1), np.nan, dtype=np.float32)

    return preds, targets, context, item_ids, metadata


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    pairs = []
    if args.model and args.dataset:
        pairs = [(args.model, args.dataset)]
    else:
        pairs = discover_pairs(args.output_dir)

    if not pairs:
        print(f"No prediction CSVs found in '{args.output_dir}'.")
        return

    for model, dataset in pairs:
        print(f"\n{'='*60}")
        print(f"Evaluating  model={model}  dataset={dataset}")
        print("="*60)

        preds, targets, context, item_ids, metadata = load_pair(
            args.output_dir, model, dataset
        )

        report = hallucination_report(
            context  = context,
            preds    = preds,
            targets  = targets,
            item_ids = item_ids,
            metadata = metadata,
        )

        # Save full report
        report_path = f"{args.output_dir}/{model}_{dataset}_report.csv"
        report.to_csv(report_path)
        print(f"  Full report → {report_path}")

        # Global summary
        print("\n  ── Global summary ──")
        print(summarise(report).to_string())

        # Per-waveform breakdown (synthetic)
        if "waveform" in report.columns:
            print("\n  ── Per waveform ──")
            print(summarise(report, group_by=["waveform"]).to_string())

        # Per-M4-group breakdown
        if "group" in report.columns:
            print("\n  ── Per M4 group ──")
            print(summarise(report, group_by=["group"]).to_string())

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model",   default=None,
                        help="Filter to a specific model name.")
    parser.add_argument("--dataset", default=None,
                        help="Filter to a specific dataset name.")
    main(parser.parse_args())