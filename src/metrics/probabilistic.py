import numpy as np
from sklearn.feature_selection import mutual_info_regression


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Compute autocorrelation function.
    x: [T] (1D signal) or [T, N] (multiple signals)
    nlags: number of lags to return
    Returns:
      [nlags+1] (if 1D) or [nlags+1, N] (if 2D)
    """
    x = np.array(x)

    if x.ndim == 1:
        x = (x - x.mean()) / (x.std() + 1e-8)
        corr = np.correlate(x, x, mode='full')
        corr = corr[corr.size // 2:]
        corr /= corr[0] + 1e-12
        return corr[:nlags+1]

    elif x.ndim == 2:
        T, N = x.shape
        out = np.zeros((nlags+1, N))
        for i in range(N):
            xi = (x[:, i] - x[:, i].mean()) / (x[:, i].std() + 1e-8)
            corr = np.correlate(xi, xi, mode='full')
            corr = corr[corr.size // 2:]
            corr /= corr[0] + 1e-12
            out[:, i] = corr[:nlags+1]
        return out

    else:
        raise ValueError(f"Unexpected x shape: {x.shape}")


import numpy as np
from sklearn.feature_selection import mutual_info_regression

def tdmi(x: np.ndarray, nlags: int, n_bins: int = 16, seed: int = 0) -> np.ndarray:
    """
    Time-Delayed Mutual Information (TDMI) via discretization.
    
    Args:
        x: np.ndarray
            [T] single signal, or
            [T, D] multivariate signal, or
            [N, T, D] batch of multivariate signals.
        nlags: int
            Number of lags to compute.
        n_bins: int
            Number of quantization bins.
        seed: int
            Random seed for MI estimation.

    Returns:
        np.ndarray
            [nlags+1] for 1D
            [nlags+1] for 2D (averaged across features)
            [N, nlags+1] for 3D batch
    """
    rng = np.random.default_rng(seed)

    # ---- Batch case ----
    if x.ndim == 3:  # [N, T, D]
        N, T, D = x.shape
        curves = []
        for i in range(N):
            curves.append(tdmi(x[i], nlags, n_bins, seed))
        return np.stack(curves, axis=0)  # [N, nlags+1]

    # ---- Multivariate case ----
    if x.ndim == 2:  # [T, D]
        T, D = x.shape
        vals = []
        for d in range(D):
            vals.append(tdmi(x[:, d], nlags, n_bins, seed))  # each [nlags+1]
        return np.mean(vals, axis=0)  # average across features

    # ---- Univariate case ----
    if x.ndim == 1:  # [T]
        x = x.astype(np.float64)
        xs = (x - x.mean()) / (x.std() + 1e-8)

        # discretize
        qs = np.quantile(xs, np.linspace(0, 1, n_bins+1))
        xsq = np.clip(np.digitize(xs, qs) - 1, 0, n_bins-1)

        res = [0.0]
        for tau in range(1, nlags+1):
            a = xsq[:-tau]
            b = xsq[tau:]
            mi = mutual_info_regression(
                a.reshape(-1, 1), b,
                discrete_features=True,
                random_state=seed
            )
            res.append(float(mi))
        return np.array(res)

    raise ValueError(f"Unexpected input shape {x.shape}")
