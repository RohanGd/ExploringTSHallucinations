
import numpy as np
from scipy.signal import stft
from scipy.stats import linregress
import statsmodels.api as sm


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


def trend_rule(context, forecast, delta=0.5):
    win_size = len(forecast)

    c_hat, p_hat = get_trend_coef(forecast)

    context_windows = rolling_windows(context, win_size)

    context_trends = []
    for w in context_windows:
        c, p = get_trend_coef(w)
        if p < 0.01:
            context_trends.append(c)

    # Case: no significant trends anywhere
    if p_hat >= 0.01 and len(context_trends) == 0:
        return True

    if p_hat < 0.01 and len(context_trends) > 0:
        diffs = [abs(c_hat / c - 1) for c in context_trends if c != 0]
        if len(diffs) == 0:
            return False
        return min(diffs) < delta

    return False



def compute_spectrum(x):
    f, t, Zxx = stft(
        x,
        window=('parzen'),
        nperseg=len(x),
        noverlap=len(x) - 1
    )
    return np.abs(Zxx).mean(axis=1)


def jaccard_distance(f1, f2):
    return 1 - (np.sum(np.minimum(f1, f2)) / np.sum(np.maximum(f1, f2)))


def frequency_rule(context, forecast, delta=0.5):
    win_size = len(forecast)

    forecast_detrended = remove_trend(forecast)
    f_hat = compute_spectrum(forecast_detrended)

    context_windows = rolling_windows(context, win_size)

    distances = []

    for w in context_windows:
        w_detrended = remove_trend(w)
        f_w = compute_spectrum(w_detrended)

        # match size
        min_len = min(len(f_hat), len(f_w))
        d = jaccard_distance(f_hat[:min_len], f_w[:min_len])
        distances.append(d)

    return min(distances) < delta


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



def fit_arma(x):
    try:
        model = sm.tsa.ARIMA(x, order=(1, 0, 1))
        res = model.fit()

        params = res.params
        pvalues = res.pvalues

        phi = params.get("ar.L1", None)
        psi = params.get("ma.L1", None)

        p_phi = pvalues.get("ar.L1", 1)
        p_psi = pvalues.get("ma.L1", 1)

        if p_phi < 0.01 and p_psi < 0.01:
            return phi, psi
    except:
        pass

    return None, None


def arma_rule(context, forecast, delta=0.5):
    context_detrended = remove_trend(context)
    forecast_detrended = remove_trend(forecast)

    phi, psi = fit_arma(context_detrended)
    phi_hat, psi_hat = fit_arma(forecast_detrended)

    if phi is None or psi is None or phi_hat is None or psi_hat is None:
        return True  # rule not applicable

    cond1 = abs(phi_hat / phi - 1) < delta if phi != 0 else False
    cond2 = abs(psi_hat / psi - 1) < delta if psi != 0 else False

    return cond1 and cond2



def knowledge_set(context, forecast):
    t = trend_rule(context, forecast)
    f = frequency_rule(context, forecast)
    p = pattern_rule(context, forecast)
    a = arma_rule(context, forecast)

    return t, f, p, a

def is_hallucination(t, f, p, a):
    # hallucination if:
    # violates trend OR frequency OR (pattern AND arma)

    if (not t) or (not f) or (not (p or a)):
        return True

    return False


import pandas as pd
import matplotlib.pyplot as plt

m4_context = pd.read_csv("outputs/chronos_m4_context.csv")
m4_forecast = pd.read_csv("outputs/chronos_m4_predictions.csv")
m4_targets = pd.read_csv("outputs/chronos_m4_targets.csv")
m4_metadata = pd.read_csv("outputs/chronos_m4_metadata.csv")

m4_metadata["forecast_trend rule"] = None
m4_metadata["forecast_frequency rule"] = None
m4_metadata["forecast_pattern rule"] = None
m4_metadata["forecast_ARMA rule"] = None
m4_metadata["target_trend rule"] = None
m4_metadata["target_frequency rule"] = None
m4_metadata["target_pattern rule"] = None
m4_metadata["target_ARMA rule"] = None
m4_metadata["forecast_hallu"] = None
m4_metadata["targets_hallu"] = None

dropna = lambda x: x[~pd.isna(x)]

for i in range(len(m4_context)):
    # Convert to numeric to handle any weird 'object' types or strings
    context  = pd.to_numeric(m4_context.iloc[i], errors='coerce').dropna().values
    forecast = pd.to_numeric(m4_forecast.iloc[i], errors='coerce').dropna().values
    targets  = pd.to_numeric(m4_targets.iloc[i], errors='coerce').dropna().values

    t, f, p, a = knowledge_set(context=context, forecast=forecast)
    m4_metadata.at[i, "forecast_trend rule"], m4_metadata.at[i, "forecast_frequency rule"], m4_metadata.at[i, "forecast_pattern rule"], m4_metadata.at[i, "forecast_ARMA rule"] = t, f, p, a
    m4_metadata.at[i, "forecast_hallu"] = is_hallucination(t, f, p, a)

    t, f, p, a = knowledge_set(context=context, forecast=targets)
    m4_metadata.at[i, "target_trend rule"], m4_metadata.at[i, "target_frequency rule"], m4_metadata.at[i, "target_pattern rule"], m4_metadata.at[i, "target_ARMA rule"] = t, f, p, a
    m4_metadata.at[i, "targets_hallu"] = is_hallucination(t, f, p, a)


    forecast, targets = forecast.tolist(), targets.tolist()
    forecast.insert(0, context[-1])
    targets.insert(0, context[-1])

    plt.figure(figsize=(12, 5))
    
    # Context (History)
    plt.plot(range(len(context)), context, label='Context (History)', color='black', linewidth=2)
    
    # Forecast (Model Prediction) - Starts where context ends
    forecast_range = range(len(context), len(context) + len(forecast))
    plt.plot(forecast_range, forecast, label='Forecast', color='red', linestyle='--')
    
    # Targets (Actual Ground Truth)
    target_range = range(len(context), len(context) + len(targets))
    plt.plot(target_range, targets, label='Actual Targets', color='blue', linestyle='--', alpha=0.6)

    plt.title(f"Time Series Comparison - Row {i}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if i == 10:
        break
m4_metadata.head(10).to_csv("targestemp.csv")