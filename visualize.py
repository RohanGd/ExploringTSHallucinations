"""
visualize.py
------------
Load saved CSVs and save one PNG per series showing context / target / prediction.

Usage
-----
# All series for a model+dataset pair
python visualize.py --model chronos --dataset synthetic

# Specific item ids
python visualize.py --model chronos --dataset m4 --ids m4_H1 m4_H2 m4_H3

# Random sample of 50 series
python visualize.py --model chronos --dataset synthetic --sample 50

# Side-by-side comparison of two models on the same dataset
python visualize.py --model chronos timesfm --dataset synthetic --sample 30

# Change output dir
python visualize.py --model chronos --dataset synthetic --plot_dir plots/synthetic
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── colour palette ────────────────────────────────────────────────────────────
C_CONTEXT = "#4C72B0"
C_TARGET  = "#2ca02c"
C_PRED    = {
    0: "#DD8452",   # model 0  (orange)
    1: "#9467bd",   # model 1  (purple)
    2: "#e377c2",   # model 2  (pink)
}
ALPHA_CONTEXT = 0.7


# ─────────────────────────────────────────────────────────────────────────────
def load_csvs(output_dir: str, model: str, dataset: str) -> dict:
    """Load all CSVs for one (model, dataset) pair into numpy arrays."""
    prefix = os.path.join(output_dir, f"{model}_{dataset}")

    # Metadata as dict list
    meta_path = f"{prefix}_metadata.csv"
    if os.path.exists(meta_path):
        mdf      = pd.read_csv(meta_path, index_col="item_id")
        metadata = {iid: row.to_dict() for iid, row in mdf.iterrows()}
    else:
        metadata = {}

    def _load(suffix):
        path = f"{prefix}_{suffix}.csv"
        if not os.path.exists(path):
            return None, None
        df = pd.read_csv(path, index_col="item_id")
        return df.index.tolist(), df.values.astype(np.float32)

    ids_p, preds   = _load("predictions")
    ids_t, targets = _load("targets")
    ids_c, context = _load("context")

    if ids_p is None:
        raise FileNotFoundError(f"No predictions CSV found at {prefix}_predictions.csv")

    return {
        "ids":      ids_p,
        "preds":    preds,
        "targets":  targets,   # may be None
        "context":  context,   # may be None
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _strip_leading_nan(arr: np.ndarray) -> np.ndarray:
    """Remove leading NaNs (left-padding artefact on context)."""
    if arr is None:
        return None
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.array([])
    return arr[np.argmax(mask):]


def _valid_target_len(tgt: np.ndarray) -> int:
    """
    Return the number of real (non-NaN) target steps.
    Targets are tail-padded → count from the right.
    """
    if tgt is None:
        return 0
    mask = ~np.isnan(tgt)
    if not mask.any():
        return 0
    # last True position + 1
    return int(np.where(mask)[0][-1]) + 1


def _meta_str(meta: dict) -> str:
    if not meta:
        return ""
    return "  |  ".join(f"{k}={v}" for k, v in meta.items())


# ─────────────────────────────────────────────────────────────────────────────
def plot_single_model(
    item_id:   str,
    data:      dict,
    model:     str,
    plot_dir:  str,
) -> None:
    """Save one PNG for one series / one model."""
    idx = data["ids"].index(item_id)

    ctx     = _strip_leading_nan(data["context"][idx]) if data["context"] is not None else None
    tgt_raw = data["targets"][idx] if data["targets"] is not None else None
    prd     = data["preds"][idx]

    # How many target steps actually have ground truth
    valid_tgt = _valid_target_len(tgt_raw)
    tgt       = tgt_raw[:valid_tgt] if tgt_raw is not None else None   # strip NaN tail

    ctx_len  = len(ctx) if ctx is not None else 0
    h_total  = len(prd)
    # x=0 is the forecast boundary — context runs on negative axis
    # This way all series align regardless of context length
    x_ctx    = np.arange(-ctx_len, 0)
    x_future = np.arange(0, h_total)

    fig, ax = plt.subplots(figsize=(14, 4))

    if ctx is not None and len(ctx):
        ax.plot(x_ctx, ctx, color=C_CONTEXT, lw=1.2, alpha=ALPHA_CONTEXT,
                label=f"Context ({ctx_len} steps)")

    if tgt is not None and len(tgt):
        ax.plot(x_future[:valid_tgt], tgt,
                color=C_TARGET, lw=2.0, label=f"Target ({valid_tgt} steps)", zorder=3)

    # Prediction: split into "has ground truth" vs "beyond target"
    ax.plot(x_future[:valid_tgt], prd[:valid_tgt],
            color=C_PRED[0], lw=2.0, linestyle="--",
            label=f"Pred ({model})", zorder=4)
    if valid_tgt < h_total:
        ax.plot(x_future[valid_tgt:], prd[valid_tgt:],
                color=C_PRED[0], lw=1.2, linestyle="--", alpha=0.35,
                label=f"Pred beyond target ({h_total - valid_tgt} steps)", zorder=4)
        # Shade the "no ground truth" zone
        ax.axvspan(valid_tgt - 0.5, h_total + 0.5,
                   color="grey", alpha=0.08, label="_nolegend_")

    ax.axvline(x=-0.5, color="grey", lw=0.8, linestyle=":")
    ax.axhline(y=0,    color="grey", lw=0.4, linestyle=":", alpha=0.4)
    ax.set_xlabel("Steps relative to forecast start  (context ← | → horizon)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_id  = item_id.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(plot_dir, f"{safe_id}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model(
    item_id:   str,
    all_data:  list[dict],
    models:    list[str],
    plot_dir:  str,
) -> None:
    """
    Save one PNG comparing N models on the same series.
    Top row: full view.  Bottom row: horizon zoom.
    """
    ref     = all_data[0]
    idx     = ref["ids"].index(item_id)
    ctx     = _strip_leading_nan(ref["context"][idx]) if ref["context"] is not None else None
    tgt_raw = ref["targets"][idx] if ref["targets"] is not None else None

    valid_tgt = _valid_target_len(tgt_raw)
    tgt       = tgt_raw[:valid_tgt] if tgt_raw is not None else None

    h_total  = len(ref["preds"][idx])
    ctx_len  = len(ctx) if ctx is not None else 0
    x_ctx    = np.arange(-ctx_len, 0)
    x_future = np.arange(0, h_total)

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1.2], hspace=0.45)
    ax_full = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])

    for ax in (ax_full, ax_zoom):
        if ctx is not None and len(ctx) and ax is ax_full:
            ax.plot(x_ctx, ctx, color=C_CONTEXT, lw=1.2,
                    alpha=ALPHA_CONTEXT, label=f"Context ({ctx_len} steps)")

        if tgt is not None and len(tgt):
            ax.plot(x_future[:valid_tgt], tgt,
                    color=C_TARGET, lw=2.2,
                    label=f"Target ({valid_tgt} steps)", zorder=3)

        for mi, (data, model) in enumerate(zip(all_data, models)):
            prd  = data["preds"][data["ids"].index(item_id)]
            col  = C_PRED[mi % len(C_PRED)]
            ax.plot(x_future[:valid_tgt], prd[:valid_tgt],
                    color=col, lw=1.8, linestyle="--",
                    label=f"Pred ({model})", zorder=4 + mi)
            if valid_tgt < h_total:
                ax.plot(x_future[valid_tgt:], prd[valid_tgt:],
                        color=col, lw=1.0, linestyle="--", alpha=0.3,
                        label="_nolegend_", zorder=4 + mi)

        if valid_tgt < h_total:
            ax.axvspan(valid_tgt - 0.5, h_total + 0.5,
                       color="grey", alpha=0.08)

        ax.axvline(x=-0.5, color="grey", lw=0.8, linestyle=":")
        ax.axhline(y=0,    color="grey", lw=0.4, linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Steps relative to forecast start  (context ← | → horizon)")

    zoom_pad = max(2, valid_tgt // 4)
    ax_zoom.set_xlim(-zoom_pad, valid_tgt + zoom_pad)
    ax_zoom.set_title("Forecast horizon (zoomed)", fontsize=8)

    meta = ref["metadata"].get(item_id, {})
    fig.suptitle(f"{item_id}    {_meta_str(meta)}", fontsize=9, y=1.01)

    safe_id  = item_id.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(plot_dir, f"{safe_id}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.plot_dir, exist_ok=True)

    # Load data for every requested model
    all_data = []
    for model in args.model:
        print(f"Loading {model}/{args.dataset} …")
        all_data.append(load_csvs(args.output_dir, model, args.dataset))

    # Common item ids across all models
    id_sets = [set(d["ids"]) for d in all_data]
    ids     = sorted(id_sets[0].intersection(*id_sets[1:]))

    if args.ids:
        ids = [i for i in args.ids if i in ids]
        missing = [i for i in args.ids if i not in ids]
        if missing:
            print(f"Warning: IDs not found: {missing}")
    elif args.sample:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(ids, size=min(args.sample, len(ids)),
                         replace=False).tolist()

    print(f"Plotting {len(ids)} series → {args.plot_dir}/")

    multi = len(args.model) > 1
    for i, item_id in enumerate(ids):
        if multi:
            plot_multi_model(item_id, all_data, args.model, args.plot_dir)
        else:
            plot_single_model(item_id, all_data[0], args.model[0], args.plot_dir)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ids)} done …")

    print(f"Done. {len(ids)} PNGs saved to {args.plot_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise context/target/prediction")
    parser.add_argument("--model",      nargs="+", required=True,
                        help="Model name(s). Multiple = side-by-side comparison.")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name (e.g. synthetic, m4).")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory containing the CSVs.")
    parser.add_argument("--plot_dir",   default=None,
                        help="Where to save PNGs (default: plots/<model>_<dataset>).")
    parser.add_argument("--ids",        nargs="+", default=None,
                        help="Specific item IDs to plot.")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Randomly sample N series.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    if args.plot_dir is None:
        model_tag    = "_vs_".join(args.model)
        args.plot_dir = os.path.join("plots", f"{model_tag}_{args.dataset}")

    main(args)