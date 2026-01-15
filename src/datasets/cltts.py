import numpy as np
from .var_utils import make_labels


# ======================================================================
# CLTTS: Cross-Lagged Time-Series (2D AR + cross-dependence) generator
# ======================================================================

def generate_cltts(
    num_samples: int = 400,
    seq_len: int = 50,
    window_size: int = 5,
    cross_coeff: float = 0.8,   # alpha
    autocorr_coeff: float = 0.5,       # gamma
    noise: float = 1.0,            # scales innovation term
    label_mode: str = "local",
    periodic_1: dict = None,
    periodic_2: dict = None,
    **kwargs
):
    """
    CLTTS dataset in the same schema as generate_var.

    Returns
    -------
    X_all : [N, T, D]  with D=2  (features: 0 -> driver, 1 -> dependent)
    y_all : [N]        labels per sequence, via make_labels(X_n, label_mode)
    A_list: list of length p=window_size, each A_k is [2, 2] lag-k coeffs

    Dynamics (ignoring sinusoids + noise):

        x1_t = ρ * Σ_k w1_k * x1_{t-k}
        x2_t = (1-γ) * [ ρ * Σ_k w2_k * x2_{t-k} ] + γ * Σ_k a_k * x1_{t-k}

    so the ground-truth VAR(W) matrices are:

        A_k = [[ ρ * w1_k,                  0                ],
               [ γ * a_k,        (1-γ) * ρ * w2_k          ]]

    which we return as A_list[k] = A_{k+1}.
    """
    D = 2
    W = int(window_size)
    if W < 1:
        raise ValueError("window_size must be >= 1")
    if seq_len <= W:
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
    base = np.exp(-np.arange(W) / W)
    base /= base.sum()

    # own AR weights for feature 0 and 1
    w1_raw = base * np.random.uniform(size=W)
    w1 = w1_raw / w1_raw.sum()

    w2_raw = base * np.random.uniform(size=W)
    w2 = w2_raw / w2_raw.sum()

    # cross weights (feature 0 -> feature 1)
    a_raw = np.random.uniform(size=W)
    a = a_raw / a_raw.sum()

    # ---- build ground-truth A_list: [p, D, D] ----
    A_list = []
    gamma = float(autocorr_coeff)
    alpha = float(cross_coeff)

    for k in range(W):  # k = 0..W-1 corresponds to lag = k+1
        A = np.zeros((D, D), dtype=float)
        # feature 0 <- feature 0 (self)
        A[0, 0] = alpha * w1[k]
        # feature 1 <- feature 0 (cross)
        A[1, 0] = gamma * a[k]
        # feature 1 <- feature 1 (self, downweighted by 1-γ)
        A[1, 1] = (1.0 - gamma) * alpha * w2[k]
        # feature 0 <- feature 1 is zero
        A_list.append(A)

    # ---- simulate sequences ----
    X_all = []
    y_all = []

    idx = np.arange(seq_len)

    for n in range(num_samples):
        # feature 0 (driver) and feature 1 (dependent)
        x1 = np.random.randn(seq_len)
        x2 = np.random.randn(seq_len)

        # --- initial trend segment for both features (like original CLTTS) ---
        for series in (x1, x2):
            trend_type = np.random.choice(["increase", "decrease"])
            if trend_type == "increase":
                series[:W] = np.linspace(0, 1, W) + 0.1 * np.random.randn(W)
            else:
                series[:W] = np.linspace(1, 0, W) + 0.1 * np.random.randn(W)

        # --- autoregressive core for each feature (without periodic yet) ---
        for t in range(W, seq_len):
            # feature 0: pure AR with weights w1
            past1 = x1[t - W:t]
            x1[t] = alpha * np.dot(w1, past1) + noise * np.random.randn() * (1.0 - alpha)

            # feature 1: base AR with weights w2
            past2 = x2[t - W:t]
            x2[t] = alpha * np.dot(w2, past2) + noise * np.random.randn() * (1.0 - alpha)

        # --- add periodic components ---
        s1 = np.zeros(seq_len)
        for period, amp in periodic_1.items():
            s1 += amp * np.sin(2.0 * np.pi * idx / float(period))
        x1 += s1

        s2 = np.zeros(seq_len)
        for period, amp in periodic_2.items():
            s2 += amp * np.sin(2.0 * np.pi * idx / float(period))
        x2 += s2

        # --- inject cross-dependence: feature 0 -> feature 1 ---
        for t in range(W, seq_len):
            past1 = x1[t - W:t]
            x2[t] = (1.0 - gamma) * x2[t] + gamma * np.dot(a, past1)

        # ---- stack into [T, D] ----
        X = np.stack([x1, x2], axis=1)  # [T, 2]

        # labels: same scheme as VAR-local (mode="local" uses feature 0)
        y = make_labels(X, mode=label_mode)

        X_all.append(X)
        y_all.append(y)

    X_all = np.stack(X_all, axis=0)  # [N, T, 2]
    y_all = np.asarray(y_all)        # [N]

    return X_all, y_all, A_list

