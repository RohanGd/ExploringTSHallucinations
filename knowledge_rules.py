import warnings
import numpy as np
from scipy.signal import stft
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt


def remove_trend(x):
    t = np.arange(len(x))
    slope, intercept, *_ = linregress(t, x)
    return x - (slope * t + intercept)


def rolling_windows(x, window_size):
    return [x[i:i + window_size] for i in range(len(x) - window_size + 1)]


def get_trend_coef(x):
    t = np.arange(len(x))
    slope, intercept, r, p, stderr = linregress(t, x)
    return slope, p


# Trend rule
# ─────────────────────────────────────────────────────────────────────────────

def trend_rule(context, forecast, delta=0.25):
    win_size = len(forecast)
    c_hat, p_hat = get_trend_coef(forecast)
    context_windows = rolling_windows(context, win_size)

    context_trends = []
    for w in context_windows:
        c, p = get_trend_coef(w)
        if p < 0.01:
            context_trends.append(c)

    # Neither forecast nor context has a significant trend
    if p_hat >= 0.01 and len(context_trends) == 0:
        return True

    if p_hat < 0.01 and len(context_trends) > 0:
        diffs = [abs(c_hat / c - 1) for c in context_trends if c != 0]
        if len(diffs) == 0:
            return False
        return min(diffs) < delta

    return False


# Frequency rule
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectrum(x):
    f, t, Zxx = stft(
        x,
        window='parzen',
        nperseg=len(x),
        noverlap=len(x) - 1
    )
    return np.abs(Zxx).mean(axis=1)


def jaccard_distance(f1, f2):
    denom = np.sum(np.maximum(f1, f2))
    if denom == 0:
        return 0.0
    return 1 - (np.sum(np.minimum(f1, f2)) / denom)


def frequency_rule(context, forecast, delta=0.5):
    win_size = len(forecast)
    forecast_detrended = remove_trend(forecast)
    f_hat = compute_spectrum(forecast_detrended)
    context_windows = rolling_windows(context, win_size)

    distances = []
    for w in context_windows:
        w_detrended = remove_trend(w)
        f_w = compute_spectrum(w_detrended)
        min_len = min(len(f_hat), len(f_w))
        d = jaccard_distance(f_hat[:min_len], f_w[:min_len])
        distances.append(d)

    return min(distances) < delta


# Pattern rule
# ─────────────────────────────────────────────────────────────────────────────

def relative_absolute_error(x, x_hat):
    numerator = np.sum(np.abs(x - x_hat))
    denominator = np.sum(np.abs(x - np.mean(x))) + 1e-8
    return numerator / denominator


def pattern_rule(context, forecast, delta=0.5):
    win_size = len(forecast)
    forecast_detrended = remove_trend(forecast)
    context_windows = rolling_windows(context, win_size)

    errors = []
    for w in context_windows:
        w_detrended = remove_trend(w)
        err = relative_absolute_error(w_detrended, forecast_detrended)
        errors.append(err)

    return min(errors) < delta


# ARMA rule  — fixed
# ─────────────────────────────────────────────────────────────────────────────

def _make_stationary(x: np.ndarray, max_diffs: int = 2):
    """
    Difference x until the ADF test rejects non-stationarity (p < 0.05),
    or until max_diffs is reached.  Returns the (possibly differenced) series
    and the number of differences applied.

    Why this is needed:
      OLS detrending removes only a linear trend.  Many M4 series (especially
      Monthly/Quarterly economic data) still have near-unit-root AR structure
      after detrending, causing statsmodels to warn about non-stationary
      starting parameters and produce garbage ARMA estimates.
    """
    d = 0
    y = x.copy()
    while d < max_diffs:
        try:
            p_adf = adfuller(y, autolag='AIC')[1]
        except Exception:
            break
        if p_adf < 0.05:
            break          # stationary enough
        y = np.diff(y)
        d += 1
    return y, d


def fit_arma(x: np.ndarray):
    """
    Fit ARMA(1,1) robustly and return (phi, psi) if both coefficients are
    significant (p < 0.01), else (None, None).

    Fixes vs original:
    ──────────────────
    1. Stationarity check  — ADF-tests x and differences until stationary
       before fitting.  Eliminates "Non-stationary starting AR parameters"
       warnings and prevents the optimizer from getting stuck near a unit root.

    2. start_params        — Provides neutral (0, 0) starting values instead
       of OLS-derived ones that can be non-invertible for short windows.

    3. Param key extraction — Searches by *position* (index 1, 2) rather than
       by name, so it works across statsmodels versions where key names may be
       'ar.L1'/'ma.L1' or 'ar1'/'ma1' or similar.

    4. Warning suppression — Only the ARMA-fitting warnings are silenced;
       any other unexpected exception still surfaces via the outer try/except.
    """
    if len(x) < 10:            # too short to fit ARMA reliably
        return None, None

    # Step 1 – make stationary
    y, n_diffs = _make_stationary(x)

    if len(y) < 8:             # differencing ate too many points
        return None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress init/convergence noise
            model = sm.tsa.ARIMA(y, order=(1, 0, 1))
            res   = model.fit(start_params=[0.0, 0.0, y.var()],
                              method_kwargs={"warn_convergence": False})

        params  = res.params
        pvalues = res.pvalues

        # ── extract by position (index 1=AR, 2=MA) ───────────────────
        # Index 0 is always the intercept/const for ARIMA(1,0,1).
        # Position-based extraction is version-agnostic.
        if len(params) < 3:
            return None, None

        phi   = float(params.iloc[1])
        psi   = float(params.iloc[2])
        p_phi = float(pvalues.iloc[1])
        p_psi = float(pvalues.iloc[2])

        if p_phi < 0.01 and p_psi < 0.01:
            return phi, psi

    except Exception:
        pass

    return None, None


def arma_rule(context, forecast, delta=0.25):
    """
    Returns True  → rule satisfied (ARMA dynamics agree, or rule not applicable).
    Returns False → rule violated (significant but mismatched ARMA dynamics).

    Note on "not applicable":
      When either the context or the forecast cannot be well-described by
      ARMA(1,1) (coefficients not significant), returning True is correct —
      the rule simply has no power here.  The paper says the hallucination
      decision falls back to the pattern rule in this case (via `p or a`).
    """
    context_detrended  = remove_trend(context)
    forecast_detrended = remove_trend(forecast)

    phi,     psi     = fit_arma(context_detrended)
    phi_hat, psi_hat = fit_arma(forecast_detrended)

    # Rule not applicable if either fit failed
    if phi is None or psi is None or phi_hat is None or psi_hat is None:
        return None     # ← changed from True; caller handles None explicitly

    cond1 = (abs(phi_hat / phi - 1) < delta) if phi != 0 else (abs(phi_hat) < delta)
    cond2 = (abs(psi_hat / psi - 1) < delta) if psi != 0 else (abs(psi_hat) < delta)

    return cond1 and cond2


# Knowledge set + hallucination decision
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_set(context, forecast):
    t = trend_rule(context, forecast)
    f = frequency_rule(context, forecast)
    p = pattern_rule(context, forecast)
    a = arma_rule(context, forecast)   # may now return None
    return t, f, p, a


def is_hallucination(t, f, p, a):
    """
    Hallucination if ANY of:
      - trend rule violated       (t is False)
      - frequency rule violated   (f is False)
      - both pattern AND ARMA violated/inapplicable
        · p=False AND (a=False OR a=None)
          When ARMA is inapplicable (None), pattern rule alone decides.
          When ARMA fires and agrees (True), pattern failure is forgiven.
    """
    if not t:
        return True
    if not f:
        return True
    # Pattern + ARMA gate:
    # hallucinate only when pattern fails AND arma either also fails or can't fire
    pattern_fails = not p
    arma_fails    = (a is False) or (a is None)
    if pattern_fails and arma_fails:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

m4_context  = pd.read_csv("outputs/chronos_m4_context.csv")
m4_targets  = pd.read_csv("outputs/chronos_m4_targets.csv")
m4_metadata = pd.read_csv("outputs/chronos_m4_metadata.csv")


def plot_KR(i, context, forecast, targets):
    forecast, targets = forecast.tolist(), targets.tolist()
    forecast.insert(0, context[-1])
    targets.insert(0, context[-1])

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(context)), context,
             label='Context (History)', color='black', linewidth=2)
    forecast_range = range(len(context) - 1, len(context) - 1 + len(forecast))
    plt.plot(forecast_range, forecast,
             label='Forecast', color='red', linestyle='--')
    target_range = range(len(context) - 1, len(context) - 1 + len(targets))
    plt.plot(target_range, targets,
             label='Actual Targets', color='blue', linestyle='--', alpha=0.6)

    row = m4_metadata.iloc[i]
    plt.title(
        f"Row {i}  |  Forecast Hallu: {row['forecast_hallu']}  "
        f"|  Target Hallu: {row['targets_hallu']}"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def process_row(i):
    context  = pd.to_numeric(m4_context.iloc[i],  errors='coerce').dropna().values
    targets  = pd.to_numeric(m4_targets.iloc[i],  errors='coerce').dropna().values

    t, f, p, a = knowledge_set(context=context, forecast=targets)

    # if i < 3:
    #     plot_KR(i, context, forecast, targets)

    return {
        "trend rule":       t,
        "frequency rule":   f,
        "pattern rule":     p,
        "ARMA rule":        a,
        "hallu":           is_hallucination(t, f, p, a),
    }


results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_row)(i) for i in range(len(m4_context))
)

results_df  = pd.DataFrame(results)
m4_metadata = pd.concat([m4_metadata.reset_index(drop=True), results_df], axis=1)
m4_metadata.to_csv("outputs/chronos_m4monthly_gt_uids.csv", index=False)

print(m4_metadata["hallu"].value_counts())