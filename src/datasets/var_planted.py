import numpy as np

def generate_var_with_A(A_list, seq_len=200, noise=0.1):
    """
    Generate VAR data given coefficient matrices A_list.
    A_list: list of numpy arrays [A^(1), A^(2), ..., A^(p)] each shape (D, D).
    seq_len: length of time series
    """
    p = len(A_list)
    D = A_list[0].shape[0]
    X = np.zeros((seq_len, D))
    for t in range(p, seq_len):
        acc = np.zeros(D)
        for k, A in enumerate(A_list, start=1):
            acc += A @ X[t-k]
        X[t] = acc + noise * np.random.randn(D)
    return X

# ==== Predefined planted setups ====

def A_local(D=5):
    A1 = np.zeros((D, D))
    for i in range(D):
        A1[i, i] = 0.8
    return [A1]

def A_nonlocal(D=5):
    A10 = np.zeros((D, D))
    A15 = np.zeros((D, D))
    A10[1, 3] = 0.7  # feature 1 depends on feature 3 at lag 10
    A15[2, 4] = 0.6  # feature 2 depends on feature 4 at lag 15
    return [np.zeros((D,D)) for _ in range(9)] + [A10] + [np.zeros((D,D)) for _ in range(4)] + [A15]

def A_mixed(D=5):
    # Local diag
    A1 = np.zeros((D, D))
    for i in range(D):
        A1[i,i] = 0.8
    # Nonlocal as above
    A10 = np.zeros((D, D)); A10[1,3] = 0.7
    A15 = np.zeros((D, D)); A15[2,4] = 0.6
    A_list = [A1] + [np.zeros((D,D)) for _ in range(8)] + [A10] + [np.zeros((D,D)) for _ in range(4)] + [A15]
    return A_list

def A_community(D=8):
    A1 = np.zeros((D,D))
    A2 = np.zeros((D,D))
    # Block 1: features 0–3 strongly coupled at lags 1–2
    for i in range(0,4):
        for j in range(0,4):
            A1[i,j] = 0.5
            A2[i,j] = 0.3
    # Block 2: features 4–7 weakly coupled at lags 4–5
    A4 = np.zeros((D,D)); A5 = np.zeros((D,D))
    for i in range(4,8):
        for j in range(4,8):
            A4[i,j] = 0.2
            A5[i,j] = 0.1
    return [A1, A2] + [np.zeros((D,D))] + [A4, A5]

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
