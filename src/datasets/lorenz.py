import numpy as np
from scipy.integrate import solve_ivp

def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

def generate_lorenz(seq_len=1000, dt=0.01):
    sol = solve_ivp(lorenz_system, [0, seq_len*dt], [1.0, 1.0, 1.0],
                    t_eval=np.arange(0, seq_len*dt, dt))
    return np.stack(sol.y, axis=1)  # shape (seq_len, 3)
