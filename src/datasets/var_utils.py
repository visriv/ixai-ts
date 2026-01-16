import numpy as np



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

def A_nonlocal(D=3, **params):
    """
    params:
        a9: scalar coeff for the edge at lag 9 (default 0.7)
        a10: scalar coeff for the edge at lag 10 (default 0.6)
        edge9: tuple (to_idx, from_idx) for the lag-9 edge (default (0,1))
        edge10: tuple (to_idx, from_idx) for the lag-10 edge (default (1,2))
    """
    a9  = _scalar(params, "a9", 0.7)
    a10 = _scalar(params, "a10", 0.6)
    i9, j9 = params.get("edge9", (0, 1))
    i10, j10 = params.get("edge10", (1, 2))

    A_list = [np.zeros((D, D)) for _ in range(10)]
    A9 = np.zeros((D, D))
    A9[i9, j9] = a9
    A10 = np.zeros((D, D))
    A10[i10, j10] = a10
    A_list[8]  = A9   # lag 9
    A_list[9] = A10  # lag 10
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




def A_cltts(seq_len, window_size, autocorr_coeff, cross_coeff,
            periodic_1=None, periodic_2=None, **kwargs):
    D = 2
    p = int(window_size)
    if p < 1:
        raise ValueError("p must be >= 1")
    if seq_len <= p:
        raise ValueError("seq_len must be > window_size")
    if D != 2:
        raise ValueError("CLTTS is defined for D=2 features")

    # ---- default periodic components if not provided ----
    if periodic_1 is None:
        # periods are in "time steps" (hours); amplitudes are scalars
        periodic_1 = {6: 0.1, 3: 0.05}
    if periodic_2 is None:
        periodic_2 = {4: 0.1, 2: 0.05}
    
    # ---- global lag weights (fixed across samples, like A_list in VAR) ----
    base = np.exp(-np.arange(p) / p)
    base /= base.sum()

    # own AR weights for feature 0 and 1
    w1_raw = base * np.random.uniform(size=p)
    w1 = w1_raw / w1_raw.sum()

    w2_raw = base * np.random.uniform(size=p)
    w2 = w2_raw / w2_raw.sum()

    # cross weights (feature 0 -> feature 1)
    a_raw = np.random.uniform(size=p)
    a = a_raw / a_raw.sum()

    # ---- build ground-truth A_list: [p, D, D] ----
    A_list = []
    gamma = float(autocorr_coeff)
    alpha = float(cross_coeff)

    for p_curr in range(p):  # k = 0..W-1 corresponds to lag = k+1
        A = np.zeros((D, D), dtype=float)
        # feature 0 <- feature 0 (self)
        A[0, 0] = alpha * w1[p_curr]
        # feature 1 <- feature 0 (cross)
        A[1, 0] = gamma * a[p_curr]
        # feature 1 <- feature 1 (self, downweighted by 1-γ)
        A[1, 1] = (1.0 - gamma) * alpha * w2[p_curr]
        # feature 0 <- feature 1 is zero
        A_list.append(A)
    return A_list

def gen_Alist(dataset_cfg, A_mode):
    """Generate the VAR coefficient matrices A_list based on dataset config."""

    ds_name = dataset_cfg["name"].lower()
    ds_name_to_A_mode = {
        "var_local": "local",
        "var_nonlocal": "nonlocal",
        "var_mixed": "mixed",
        "var_community": "community",
        "cltts": "cltts"}
    
    A_mode = ds_name_to_A_mode.get(ds_name, A_mode)
    params = dataset_cfg#.get("var_params", {})
    D = dataset_cfg.get("num_features", 3)

    if A_mode == "local":
        A_list = A_local(D=D, **params)
    elif A_mode == "nonlocal":
        A_list = A_nonlocal(D=D, **params)
    elif A_mode == "mixed":
        A_list = A_mixed(D=D, **params)
    elif A_mode == "community":
        A_list = A_community(D=D, **params)
    elif A_mode == "cltts":
        A_list = A_cltts(**params)
    else:
        raise ValueError(f"Unknown dataset A_mode '{A_mode}' for VAR generation.")
    return A_list


def make_labels(X, mode="local", params=None):
    """
    Produce labels aligned with the planted interaction structure.
    X: [T, D]
    mode: dataset type
    """
    T, D = X.shape
    if mode == "local":
        # Label = sign of mean of feature 0 over last 20 steps
        y = int(X[-10:,0].mean() > 0)
    elif mode == "nonlocal":
        # Label depends on long-range: feature 1 now vs feature 3 lagged by 10
        # y = int((X[-1,1] - X[-11,3]) > 0)
        y = int(X[-10:,0].mean() > 0)
    elif mode == "cltts_trend":
        delta = params["delta"]
        recent_mean = X[-delta:, 0].mean()
        prev_mean = X[-2*delta:-delta, 0].mean()
        y = int((recent_mean - prev_mean) > 0)
    elif mode == "cltts_mean":
        y = int(X[-10:,0].mean() > 0)
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
