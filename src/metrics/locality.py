import numpy as np
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit

def exp_decay(tau, a, b): return a * np.exp(-b * tau)
def power_decay(tau, a, p): return a * np.power(tau+1, -p)






def aggregate_lag_curve(lag_dict, tau_max, reduce="median"):
    """
    Aggregate lag_dict into lag curves.
    Args:
      lag_dict: {(tau, d, d'): np.ndarray[N]}
      tau_max: maximum tau
      reduce: 'median' | 'mean' (how to reduce across pairs)
    Returns:
      curve: np.ndarray[tau_max+1, N]  # per-sample lag curve
    """
    # Find N from one entry
    first_val = next(iter(lag_dict.values()))
    N = first_val.shape[0]

    curve = np.zeros((tau_max + 1, N))
    for tau in range(tau_max + 1):
        vals = [np.abs(v) for (tau_, d, d2), v in lag_dict.items() if tau_ == tau]  # list of [N]
        if vals:
            vals = np.stack(vals, axis=0)  # [num_pairs, N]
            if reduce == "median":
                curve[tau] = np.median(vals, axis=0)  # [N]
            else:
                curve[tau] = np.mean(vals, axis=0)    # [N]
        else:
            curve[tau] = 0.0
    return curve  # [tau_max+1, N]


def half_range(curve):
    """
    Half-range per sample: smallest τ where cumulative mass >= 0.5.
    Args:
      curve: np.ndarray of shape [tau_max+1] or [tau_max+1, N]
    Returns:
      np.ndarray[N] if multi-sample, or scalar np.ndarray[1] if single curve
    """
    arr = np.array(curve)
    if arr.ndim == 1:
        arr = arr[:, None]  # -> [T, 1]

    T, N = arr.shape
    res = []
    for n in range(N):
        s = arr[:, n].sum() + 1e-12
        cs = np.cumsum(arr[:, n]) / s
        res.append(int(np.searchsorted(cs, 0.5)))
    return np.array(res)


def loc_at_k(curve, K):
    """
    Loc@K: Fraction of interaction mass within first K lags.
    curve: [tau_max+1]  (aggregated interaction values per lag)
    """
    num = curve[:K+1].sum()
    denom = curve.sum() + 1e-12
    return float(num / denom)


def loc_at_50(curve):
    """
    Loc@50: Lag cutoff containing 50% of interaction mass.
    curve: [tau_max+1]
    """
    total = curve.sum() + 1e-12
    cs = np.cumsum(curve) / total
    tau50 = int(np.searchsorted(cs, 0.5))
    return tau50


def fit_decay(curve):
    """
    Fit exponential and power-law decays per-sample.
    Args:
      curve: np.ndarray[tau_max+1, N]
    Returns:
      exp_params: np.ndarray[N, 2]   # (a, b) per sample
      pow_params: np.ndarray[N, 2]   # (a, p) per sample
    """
    taus = np.arange(curve.shape[0])
    N = curve.shape[1]

    exp_params, pow_params = [], []
    for n in range(N):
        y = np.maximum(curve[:, n], 1e-12)  # avoid zeros
        try:
            exp_fit, _ = curve_fit(exp_decay, taus, y, maxfev=8000)
        except RuntimeError:
            exp_fit = [np.nan, np.nan]
        try:
            pow_fit, _ = curve_fit(power_decay, taus, y, maxfev=8000)
        except RuntimeError:
            pow_fit = [np.nan, np.nan]
        exp_params.append(exp_fit)
        pow_params.append(pow_fit)

    return np.array(exp_params), np.array(pow_params)