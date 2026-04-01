"""
M4 dataset wrapper.

Downloads the official M4 train/test CSVs directly from the M4
competition GitHub repository, bypassing datasetsforecast entirely.
This gives us full control over the pandas parser so we can handle
malformed rows (EOF-inside-string, stray quotes, etc.) that crash the
default C parser.

Cache behaviour
---------------
CSVs are downloaded once into `data_dir` and re-used on subsequent
runs.  Pass `force_download=True` to overwrite.

Strategy
--------
  context : last min(context_len, len(train_series)) points of training data.
  target  : first forecast_horizon points of the held-out test series.
            Tail-padded with NaN when the M4 horizon < forecast_horizon.

Series with fewer than `min_context_len` training points are skipped.
"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .base import TimeSeriesSample

# All six M4 groups and their official competition forecast horizons
M4_GROUPS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]

M4_NATIVE_HORIZONS = {
    "Yearly":    6,
    "Quarterly": 8,
    "Monthly":   18,
    "Weekly":    13,
    "Daily":     14,
    "Hourly":    48,
}


def _find_csv(data_dir: str, group: str, split: str) -> str:
    """
    Locate an M4 CSV on disk, checking every known local layout.

    Layouts probed (in order):
      1. {data_dir}/m4/datasets/{Group}-{split}.csv   ← datasetsforecast default
      2. {data_dir}/{Group}-{split}.csv               ← flat
      3. {data_dir}/M4-{Group}-{split}.csv            ← our old downloader
      4. {data_dir}/{Split}/{Group}-{split}.csv        ← mirrors GitHub structure

    With data_dir="data/m4", layout 1 resolves to:
        data/m4/m4/datasets/Monthly-train.csv
    which matches the user's existing files.
    """
    split_cap = split.capitalize()
    candidates = [
        os.path.join(data_dir, "m4", "datasets", f"{group}-{split}.csv"),
        os.path.join(data_dir, f"{group}-{split}.csv"),
        os.path.join(data_dir, f"M4-{group}-{split}.csv"),
        os.path.join(data_dir, split_cap, f"{group}-{split}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find M4 {group}/{split} CSV.  Looked in:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + f"\n\nDownload from: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset"
        + f"\nand place the CSVs under {data_dir}."
    )


def _read_m4_csv(path: str) -> pd.DataFrame:
    """
    Read an M4 CSV robustly.

    M4 CSVs have the format:
        V1,V2,V3,...
        H1,1.0,2.0,...
        H2,3.0,4.0,...

    where V1 is the series ID and V2… are the time-ordered values.
    Some rows contain stray quotes or other characters that crash
    pandas' C engine → we fall back to the Python engine.

    Returns a DataFrame with columns ["unique_id", "y"] in long format.
    """
    read_kwargs = dict(
        header      = 0,
        index_col   = 0,       # first column = series ID
        dtype       = str,     # read everything as str first
        on_bad_lines = "skip", # skip malformed rows, warn below
    )

    # Try fast C engine first; fall back to Python engine on parse errors
    try:
        df = pd.read_csv(path, engine="c", **read_kwargs)
    except Exception:
        df = pd.read_csv(path, engine="python", **read_kwargs)

    # Convert to long format: index=uid, columns=t1,t2,...  → uid, y
    df.index.name = "unique_id"
    df = df.reset_index()
    df = df.melt(id_vars="unique_id", value_name="y").dropna(subset=["y"])

    # Parse values; coerce bad strings to NaN then drop them
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    # Sort so values are in original time order (columns were t1, t2, …)
    df["_order"] = df["variable"].str.extract(r"(\d+)$")[0].astype(int)
    df = df.sort_values(["unique_id", "_order"]).drop(columns=["variable", "_order"])
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
def _load_group(
    group:            str,
    data_dir:         str,
    context_len:      int,
    forecast_horizon: int | None,   # None → use M4_NATIVE_HORIZONS[group]
    min_context_len:  int,
    force_download:   bool = False,
) -> list:
    """Locate CSVs on disk, parse them, and return TimeSeriesSamples."""
    native_horizon   = M4_NATIVE_HORIZONS[group]
    forecast_horizon = forecast_horizon or native_horizon

    print(f"  Loading M4/{group}  "
          f"(native_horizon={native_horizon}, using={forecast_horizon}) …", flush=True)

    train_path = _find_csv(data_dir, group, "train")
    test_path  = _find_csv(data_dir, group, "test")

    train_df = _read_m4_csv(train_path)
    test_df  = _read_m4_csv(test_path)

    train_grp = train_df.groupby("unique_id")["y"].apply(np.array)
    test_grp  = test_df.groupby("unique_id")["y"].apply(np.array)

    samples = []
    for uid in train_grp.index:
        train_vals = train_grp[uid].astype(np.float32)
        if len(train_vals) < min_context_len:
            continue

        context = train_vals[-context_len:]

        test_vals = test_grp[uid].astype(np.float32) if uid in test_grp.index \
                    else np.array([], dtype=np.float32)

        if len(test_vals) >= forecast_horizon:
            target = test_vals[:forecast_horizon]
        else:
            # Tail-pad with NaN only when forecast_horizon > native_horizon
            target = np.full(forecast_horizon, np.nan, dtype=np.float32)
            target[: len(test_vals)] = test_vals

        samples.append(TimeSeriesSample(
            context  = context,
            target   = target,
            item_id  = f"m4_{uid}",
            metadata = {
                "group":          group,
                "uid":            uid,
                "native_horizon": native_horizon,
            },
        ))

    print(f"    {len(samples)} series kept.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
class M4Dataset(Dataset):
    """
    M4 dataset for one or more frequency groups.

    Parameters
    ----------
    groups           : groups to include; None → all six.
    context_len      : cap on context length (series shorter than this are
                       used as-is; NaN-padding is done in the collate_fn).
    forecast_horizon : target horizon per group.
                       None (default) → each group uses its native M4 horizon.
                       int            → override for all groups (use with care;
                                        e.g. 64 causes NaN-padding for all groups
                                        except Hourly).
    data_dir         : local directory for cached CSVs.
    min_context_len  : series with fewer training points are skipped.
    force_download   : re-download even if CSVs already exist.

    Attributes
    ----------
    forecast_horizon : int  — the horizon used (native if not overridden,
                               or the single override value).
                               When groups have *different* native horizons and
                               no override is set, use per-group datasets instead
                               (see run.py for the recommended pattern).
    """

    def __init__(
        self,
        groups:           list | None = None,
        context_len:      int  = 500,
        forecast_horizon: int | None = None,   # None → native per group
        data_dir:         str  = "data/m4",
        min_context_len:  int  = 10,
        force_download:   bool = False,
        filter:           str | None = None,   # path to knowledge-rule CSV
    ):
        self.context_len = context_len
        self.groups      = groups or M4_GROUPS

        # Resolve effective horizon
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon
        elif len(self.groups) == 1:
            self.forecast_horizon = M4_NATIVE_HORIZONS[self.groups[0]]
        else:
            horizons = [M4_NATIVE_HORIZONS[g] for g in self.groups]
            if len(set(horizons)) > 1:
                print(
                    f"  Warning: groups {self.groups} have different native horizons "
                    f"{horizons}. Targets will be NaN-padded to max={max(horizons)}. "
                    f"Consider running one group at a time for clean targets."
                )
            self.forecast_horizon = max(horizons)

        # ── Build allowed-id set from filter file ─────────────────────
        # CSV columns: item_id, group, uid, trend rule, frequency rule,
        #              pattern rule, ARMA rule, hallu
        # Keep only rows where hallu == False (ground truth is clean).
        allowed_ids: set | None = None
        if filter is not None:
            filter_df = pd.read_csv(filter)
            hallu_col = filter_df["hallu"]
            if hallu_col.dtype == object:
                hallu_col = hallu_col.str.strip().str.lower().map(
                    {"false": False, "true": True, "0": False, "1": True}
                )
            not_hallu  = filter_df[~hallu_col.astype(bool)]
            allowed_ids = set(not_hallu["item_id"].astype(str))
            print(f"  Filter: {len(allowed_ids)} non-hallucinated series "
                  f"loaded from {filter}")

        print("Loading M4 dataset …")
        self.samples: list[TimeSeriesSample] = []
        for g in self.groups:
            self.samples.extend(
                _load_group(g, data_dir, context_len, self.forecast_horizon,
                            min_context_len, force_download)
            )

        # Apply filter — item_id format is "m4_{uid}"
        if allowed_ids is not None:
            before = len(self.samples)
            self.samples = [s for s in self.samples if s.item_id in allowed_ids]
            print(f"  Filter applied: {before} → {len(self.samples)} series kept.")

        print(f"M4 total: {len(self.samples)} series  |  "
              f"forecast_horizon={self.forecast_horizon}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TimeSeriesSample:
        return self.samples[idx]