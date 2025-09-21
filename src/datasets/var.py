import numpy as np


def generate_var(num_samples=1000, num_series=3, seq_len=100,
                 coeff_scale=0.5, noise=0.1, label_rule="last", **kwargs):
    """
    Generate N samples of VAR sequences, each shape (T, D).
    
    Returns:
        X: [N, T, D]  (synthetic sequences)
        y: [N]        (labels per sequence)
        A: [D, D]     (coefficient matrix, shared across samples)
    """
    A = coeff_scale * np.random.randn(num_series, num_series)
    X_all, y_all = [], []

    for n in range(num_samples):
        X = np.zeros((seq_len, num_series))
        for t in range(1, seq_len):
            X[t] = A @ X[t-1] + noise * np.random.randn(num_series)
        X_all.append(X)

        # --- Label per sequence ---
        y = int(X[-1, 0] > 0)
    
        y_all.append(y)

    X_all = np.stack(X_all, axis=0)   # [N, T, D]
    y_all = np.array(y_all)           # [N]
    return X_all, y_all, A
