import numpy as np

from var_utils import A_local, A_nonlocal, A_mixed, A_community, make_labels

# ground-truth plant registry
PREDEFINED = {
    "local": A_local,
    "nonlocal": A_nonlocal,
    "mixed": A_mixed,
    "community": A_community
}

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








# def planted_relevance(A_list, seq_len=None):
#     """
#     Construct ground-truth interaction relevance tensor.
    
#     Args:
#         A_list: list of [D, D] matrices
#         seq_len: optional (to expand relevance into [T,D])
    
#     Returns:
#         R_edges: [p, D, D] absolute coefficients (lag-based relevance)
#         R_unrolled: [T, D] relevance map over sequence (optional)
#     """
#     p = len(A_list)
#     D = A_list[0].shape[0]
    
#     # interaction tensor
#     R_edges = np.zeros((p, D, D))
#     for tau, A in enumerate(A_list, start=1):
#         R_edges[tau-1] = np.abs(A)
    
#     if seq_len is not None:
#         # unroll into [T, D] by attributing past points
#         R_unrolled = np.zeros((seq_len, D))
#         for tau in range(1, p+1):
#             R_unrolled[tau:, :] += np.abs(A_list[tau-1]).sum(axis=0)[None, :]
#         return R_edges, R_unrolled
    
#     return R_edges
