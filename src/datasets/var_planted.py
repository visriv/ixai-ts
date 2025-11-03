import numpy as np

# def generate_var(A_list, seq_len=200, noise=0.1):
#     """
#     Generate VAR data given coefficient matrices A_list.
#     A_list: list of numpy arrays [A^(1), A^(2), ..., A^(p)] each shape (D, D).
#     seq_len: length of time series
#     """
#     p = len(A_list)
#     D = A_list[0].shape[0]
#     X = np.zeros((seq_len, D))
#     for t in range(p, seq_len):
#         acc = np.zeros(D)
#         for k, A in enumerate(A_list, start=1):
#             acc += A @ X[t-k]
#         X[t] = acc + noise * np.random.randn(D)
#     return X


import numpy as np

def generate_var(num_samples=1000, num_series=3, seq_len=100,
                 noise=0.1, mode='local', **A_params):
    """
    Generate N samples of VAR sequences, each shape (T, D).

    Returns:
        X: [N, T, D]      synthetic sequences
        y: [N]            labels per sequence (via make_labels)
        A_list: list[np.ndarray] of length p, where A_list[k-1] is lag-k coeff (DxD)
    """
    # --- choose lag matrices ---
    if mode == 'local':
        A_list = A_local(num_series, **A_params)
    elif mode == 'nonlocal':
        A_list = A_nonlocal(num_series, **A_params)
    elif mode == 'mixed':
        A_list = A_mixed(num_series, **A_params)
    elif mode == 'community':
        A_list = A_community(num_series, **A_params)
    else:
        raise ValueError(f"unknown mode: {mode}")

    p = len(A_list)
    if p < 1:
        raise ValueError("A_list must contain at least one lag matrix")

    X_all, y_all = [], []

    for n in range(num_samples):
        # start from zeros; no burn-in / intercept
        X = np.zeros((seq_len, num_series), dtype=float)

        # main VAR recursion:
        # x_t = sum_{k=1..min(p,t)} A_k x_{t-k} + eps_t
        for t in range(1, seq_len):
            xt = np.zeros(num_series, dtype=float)
            max_k = min(p, t)
            for k in range(1, max_k + 1):
                xt += A_list[k-1] @ X[t - k]
            X[t] = xt + noise * np.random.randn(num_series)

        X_all.append(X)
        y_all.append(make_labels(X, mode))

    X_all = np.stack(X_all, axis=0)  # [N, T, D]
    y_all = np.asarray(y_all)        # [N]
    return X_all, y_all, A_list


# ==== Predefined planted setups ====
import numpy as np

# ---- small helper ----
def _scalar(params, key, default):
    """Return scalar param if provided, else default."""
    val = params.get(key, default)
    if np.ndim(val) == 0:
        return float(val)
    raise ValueError(f"'{key}' must be a scalar; got shape {np.shape(val)}")

# ----------------- VAR lag designs with param overrides -----------------

def A_local(D=5, **params):
    """
    params:
        a1: scalar for the diagonal self-dependence at lag 1 (default 0.8)
    """
    a1 = _scalar(params, "a1", 0.8)
    A1 = np.zeros((D, D))
    np.fill_diagonal(A1, a1)
    return [A1]

def A_nonlocal(D=5, **params):
    """
    params:
        a10: scalar coeff for the edge at lag 10 (default 0.7)
        a15: scalar coeff for the edge at lag 15 (default 0.6)
        edge10: tuple (to_idx, from_idx) for the lag-10 edge (default (1,3))
        edge15: tuple (to_idx, from_idx) for the lag-15 edge (default (2,4))
    """
    a10  = _scalar(params, "a10", 0.7)
    a15  = _scalar(params, "a15", 0.6)
    i10, j10 = params.get("edge10", (1, 3))
    i15, j15 = params.get("edge15", (2, 4))

    A_list = [np.zeros((D, D)) for _ in range(15)]
    A10 = np.zeros((D, D)); A10[i10, j10] = a10
    A15 = np.zeros((D, D)); A15[i15, j15] = a15
    A_list[9]  = A10  # lag 10
    A_list[14] = A15  # lag 15
    return A_list

def A_mixed(D=5, **params):
    """
    params:
        a1:  scalar for diagonal self-dependence at lag 1 (default 0.8)
        a10: scalar for nonlocal edge at lag 10 (default 0.7)
        a15: scalar for nonlocal edge at lag 15 (default 0.6)
        edge10: tuple (to_idx, from_idx) (default (1,3))
        edge15: tuple (to_idx, from_idx) (default (2,4))
    """
    a1  = _scalar(params, "a1", 0.8)
    a10 = _scalar(params, "a10", 0.7)
    a15 = _scalar(params, "a15", 0.6)
    i10, j10 = params.get("edge10", (1, 3))
    i15, j15 = params.get("edge15", (2, 4))

    A_list = [np.zeros((D, D)) for _ in range(15)]

    A1 = np.zeros((D, D)); np.fill_diagonal(A1, a1)
    A10 = np.zeros((D, D)); A10[i10, j10] = a10
    A15 = np.zeros((D, D)); A15[i15, j15] = a15

    A_list[0]  = A1   # lag 1
    A_list[9]  = A10  # lag 10
    A_list[14] = A15  # lag 15
    return A_list

def A_community(D=8, **params):
    """
    params:
        a1, a2, a4, a5: scalars for the block strengths at lags 1,2,4,5
                        (defaults: 0.5, 0.3, 0.2, 0.1 respectively)
        block1: (start, end) indices for community 1  (default (0,4))
        block2: (start, end) indices for community 2  (default (4,8))
    """
    a1 = _scalar(params, "a1", 0.5)
    a2 = _scalar(params, "a2", 0.3)
    a4 = _scalar(params, "a4", 0.2)
    a5 = _scalar(params, "a5", 0.1)

    b1s, b1e = params.get("block1", (0, 4))
    b2s, b2e = params.get("block2", (4, 8))

    p = 5
    A_list = [np.zeros((D, D)) for _ in range(p)]

    A1 = np.zeros((D, D)); A2 = np.zeros((D, D))
    A4 = np.zeros((D, D)); A5 = np.zeros((D, D))

    # block 1 at lags 1–2
    A1[b1s:b1e, b1s:b1e] = a1
    A2[b1s:b1e, b1s:b1e] = a2
    # block 2 at lags 4–5
    A4[b2s:b2e, b2s:b2e] = a4
    A5[b2s:b2e, b2s:b2e] = a5

    A_list[0] = A1   # lag 1
    A_list[1] = A2   # lag 2
    A_list[3] = A4   # lag 4
    A_list[4] = A5   # lag 5
    return A_list


# ground-truth plant registry
PREDEFINED = {
    "local": A_local,
    "nonlocal": A_nonlocal,
    "mixed": A_mixed,
    "community": A_community
}

def make_labels(X, mode="local"):
    """
    Produce labels aligned with the planted interaction structure.
    X: [T, D]
    mode: dataset type
    """
    T, D = X.shape
    if mode == "local":
        # Label = sign of mean of feature 0 over last 20 steps
        y = int(X[-20:,0].mean() > 0)
    elif mode == "nonlocal":
        # Label depends on long-range: feature 1 now vs feature 3 lagged by 10
        y = int((X[-1,1] - X[-11,3]) > 0)
    elif mode == "mixed":
        # Combination: feature 0 local + feature 3 nonlocal
        y = int((X[-1,0] + X[-11,3]) > 0)
    elif mode == "community":
        # Which block dominates energy
        block1 = (X[-50:,0:4]**2).sum()
        block2 = (X[-50:,4:8]**2).sum()
        y = int(block1 > block2)
    else:
        raise ValueError
    return y


def planted_relevance(A_list, seq_len=None):
    """
    Construct ground-truth interaction relevance tensor.
    
    Args:
        A_list: list of [D, D] matrices
        seq_len: optional (to expand relevance into [T,D])
    
    Returns:
        R_edges: [p, D, D] absolute coefficients (lag-based relevance)
        R_unrolled: [T, D] relevance map over sequence (optional)
    """
    p = len(A_list)
    D = A_list[0].shape[0]
    
    # interaction tensor
    R_edges = np.zeros((p, D, D))
    for tau, A in enumerate(A_list, start=1):
        R_edges[tau-1] = np.abs(A)
    
    if seq_len is not None:
        # unroll into [T, D] by attributing past points
        R_unrolled = np.zeros((seq_len, D))
        for tau in range(1, p+1):
            R_unrolled[tau:, :] += np.abs(A_list[tau-1]).sum(axis=0)[None, :]
        return R_edges, R_unrolled
    
    return R_edges
