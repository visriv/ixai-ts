import numpy as np
from typing import Tuple, Dict, Sequence


def _lorenz_rhs(state: np.ndarray,
                sigma: float,
                beta: float,
                rho: float) -> np.ndarray:
    """
    Lorenz-63 RHS:
        x' = sigma (y - x)
        y' = x (rho - z) - y
        z' = x y - beta z

    state: (3,) array [x, y, z]
    returns: (3,) array [dx, dy, dz]
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


def _simulate_lorenz_trajectory(rho: float,
                                seq_len: int,
                                dt: float,
                                sigma: float,
                                beta: float,
                                noise: float,
                                init_box: Sequence[float]) -> np.ndarray:
    """
    Simulate one Lorenz trajectory with simple Euler integration.

    Returns:
        X: [seq_len, 3] with columns [x, y, z].
    """
    low, high = init_box
    x0 = np.random.uniform(low, high, size=3)

    X = np.zeros((seq_len, 3), dtype=float)
    X[0] = x0

    for t in range(1, seq_len):
        dx = _lorenz_rhs(X[t - 1], sigma=sigma, beta=beta, rho=rho)
        X[t] = X[t - 1] + dt * dx
        if noise > 0.0:
            X[t] += noise * np.random.randn(3)

    return X


def _label_stability_lorenz(X: np.ndarray,
                            sigma: float,
                            beta: float,
                            rho: float,
                            window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given one Lorenz trajectory X of shape [T, 3],
    compute:

      - features F: [T, 6] = [x, y, z, dx, dy, dz]
      - labels y:  [T] with
            0 -> stable
            1 -> unstable

    Label rule (from the paper):

      1. Compute mean(x) over the trajectory.
      2. Regime r_t = +1 if x_t > mean_x, else -1.
      3. y_t = 0 (stable) iff the past `window` and future `window`
         neighbours all have the same regime as r_t.
         Otherwise y_t = 1 (unstable).
      4. Boundary points without a full window are marked unstable.
    """
    T = X.shape[0]
    # compute derivatives at each time using the ODE (not finite differences)
    dX = np.zeros_like(X)
    for t in range(T):
        dX[t] = _lorenz_rhs(X[t], sigma=sigma, beta=beta, rho=rho)

    # 6-D features per time step
    F = np.concatenate([X, dX], axis=1)   # [T, 6]

    x = X[:, 0]
    mean_x = np.mean(x)
    r = np.where(x > mean_x, 1, -1)      # regime sequence

    y = np.ones(T, dtype=int)            # default: unstable (1)
    for t in range(window, T - window):
        if np.all(r[t - window:t + window + 1] == r[t]):
            y[t] = 0                     # stable

    return F, y


def generate_lorenz(
    num_samples: int,
    seq_len: int,
    dt: float = 0.01,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    rho_values: Sequence[float] = (28.0,),
    task: str = "classification",  # currently only classification really makes sense
    noise: float = 0.0,
    init_box: Sequence[float] = (-1.0, 1.0),
    all_times: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate N samples of Lorenz data, following the setup in the paper.

    For each sample n:
      - draw rho_n from `rho_values`
      - simulate one trajectory X_n[t, :] = [x_t, y_t, z_t], t=0..T-1
      - compute features F_n[t, :] = [x_t, y_t, z_t, dx_t, dy_t, dz_t]
      - assign label y_n[t] ∈ {0,1} (stable/unstable) per the window rule.

    Returns:
        X_all  : [N, T, 6]  feature sequences
        y_all  : [N, T]     0 = stable, 1 = unstable
        params : dict with:
                  - "sigma": scalar
                  - "beta" : scalar
                  - "rho"  : [N] array of rho used per sample
    """
    rho_values = np.asarray(rho_values, dtype=float)
    if rho_values.ndim != 1 or rho_values.size < 1:
        raise ValueError("rho_values must be a 1D sequence with at least one element")

    X_all = []
    y_all = []
    rho_all = []

    if task != "classification":
        raise ValueError("Only 'classification' (stable vs unstable) is supported here.")

    num_rhos = rho_values.size

    for n in range(num_samples):
        # pick a rho for this trajectory (uniform over supplied values)
        rho = float(rho_values[np.random.randint(0, num_rhos)])

        # simulate trajectory and build features + labels
        X = _simulate_lorenz_trajectory(
            rho=rho,
            seq_len=seq_len,
            dt=dt,
            sigma=sigma,
            beta=beta,
            noise=noise,
            init_box=init_box,
        )
        F, y_seq = _label_stability_lorenz(X, sigma=sigma, beta=beta, rho=rho, window=5)

        X_all.append(F)       # [T, 6]
        y_all.append(y_seq)   # [T]
        rho_all.append(rho)

    X_all = np.stack(X_all, axis=0)             # [N, T, 6]
    y_all = np.stack(y_all, axis=0)             # [N, T]
    rho_all = np.asarray(rho_all, dtype=float)  # [N]

    params = {
        "sigma": float(sigma),
        "beta": float(beta),
        "rho": rho_all,
    }

    return X_all, y_all, params
