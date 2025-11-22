import torch
import numpy as np
from typing import Dict, Tuple, Optional, Callable


# ------------------------------
# 1. Scalar output helper
# ------------------------------
def _scalar_output(model: torch.nn.Module,
                   x: torch.Tensor,
                   target_idx: Optional[int] = None) -> torch.Tensor:
    """
    Given input x and a model, return a scalar F(x) to attribute.

    x: [1, ...] tensor
    If model(x) is [1, C], we either pick target_idx (if given) or argmax.
    If model(x) is scalar, we just squeeze.
    """
    logits = model(x)  # [1, C] or scalar
    if logits.ndim == 2:
        if target_idx is None:
            target_idx = logits.argmax(dim=1).item()
        return logits[0, target_idx]
    else:
        return logits.squeeze()


# ------------------------------
# 2. Base IH on a flattened input
# ------------------------------
def integrated_hessians_flat(
    model: torch.nn.Module,
    x_vec: torch.Tensor,        # [D]
    baseline_vec: torch.Tensor, # [D]
    m_steps: int = 50,
    target_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Integrated Hessians for a single example with a flattened input.

    Implements (for features i, j):

        IH_ij(x) = (x_i - x'_i)(x_j - x'_j) * ∫_0^1 H_ij(x' + α (x - x')) dα

    where H_ij is the Hessian of a chosen scalar F wrt input features.

    Returns:
        IH_flat: [D, D] tensor.
    """
    model.eval()

    # Ensure 1D vectors
    x_vec = x_vec.detach().view(-1)
    baseline_vec = baseline_vec.detach().view(-1)
    delta = x_vec - baseline_vec                 # [D]
    D = delta.shape[0]

    # Predefine f: R^D -> scalar
    def f(z_flat: torch.Tensor) -> torch.Tensor:
        """
        z_flat: [D]
        """
        # Restore batch dimension so model sees [1, ...F...]
        z_1 = z_flat.view(1, -1)
        return _scalar_output(model, z_1, target_idx=target_idx)

    # Integration nodes in (0, 1]; skip α=0 for stability
    alphas = torch.linspace(0.0, 1.0, steps=m_steps + 1, device=x_vec.device)[1:]

    H_sum = torch.zeros(F, F, device=x_vec.device)

    for alpha in alphas:
        x_alpha = baseline_vec + alpha * delta  # [D]

        # Full Hessian wrt x_alpha (shape [D, D])
        H = torch.autograd.functional.hessian(f, x_alpha)  # [D, D]
        H_sum += H

    H_avg = H_sum / alphas.numel()  # numerical approximation of path integral

    # Feature scaling: (Δx_i Δx_j)
    d = delta.view(-1)                       # [D]
    IH_flat = H_avg * (d.view(D, 1) * d.view(1, D))  # [D, D]

    return IH_flat.detach()


# ------------------------------
# 3. Time-series IH with lag dict
# ------------------------------
def integrated_hessians_timeseries(
    model: torch.nn.Module,
    x: torch.Tensor,                   # [B, T, D]
    baseline: torch.Tensor,            # [1, T, D] or [B, T, D]
    m_steps: int = 50,
    target_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int, int, int], np.ndarray]]:
    """
    Integrated Hessians for time-series inputs.

    Args:
        model     : PyTorch model mapping [B, T, D] -> [B, C] or [B, 1]
        x         : [B, T, D] input batch
        baseline  : [1, T, D] or [B, T, D]
        m_steps   : # of integration points α in (0,1]
        target_idx: class / logit index for scalar F; if None, argmax is used per sample

    Returns:
        ih_feat_matrix: [B, D, D]
            - feature–feature interactions aggregated over both time axes (mean).
        ih_full_ts   : [B, T, D, T, D]
            - full per-time, per-feature interaction tensor.
        ih_lag_dict  : dict mapping
            (t, d, d2, tau) -> np.ndarray of shape [B]
            - same style as your STI dict, so you can feed it directly
              into your existing aggregation code:
                  agg.setdefault((tau, d, d2), []).append(arr)
    """
    model.eval()
    device = x.device

    B, T, D = x.shape
    F = T * D

    # Prepare baseline broadcast
    if baseline.shape[0] == 1:
        baseline = baseline.expand(B, T, D)  # [B, T, D]
    elif baseline.shape[0] != B:
        raise ValueError(f"baseline batch dim {baseline.shape[0]} != 1 or B={B}")

    baseline = baseline.to(device)
    x = x.to(device)

    ih_full_list = []  # each element: [T, D, T, D]

    for b in range(B):
        x_b = x[b].reshape(-1)          # [D]
        base_b = baseline[b].reshape(-1) # [D]

        IH_flat_b = integrated_hessians_flat(
            model=model,
            x_vec=x_b,
            baseline_vec=base_b,
            m_steps=m_steps,
            target_idx=target_idx,
        )  # [D, D]

        # Reshape to time-series structure [T, D, T, D]
        IH_ts_b = IH_flat_b.view(T, D, T, D)  # first (t,d), then (t',d2)
        ih_full_list.append(IH_ts_b)

    # Stack over batch: [B, T, D, T, D]
    ih_full_ts = torch.stack(ih_full_list, dim=0)  # [B, T, D, T, D]

    # ------------------------------
    # 3a. Aggregate to [B, D, D]
    # ------------------------------
    # Mean over both time axes (anchor t and interacting t'):
    # ih_feat_matrix[b, d, d2] = mean_t,t' IH[b, t, d, t', d2]
    ih_feat_matrix = ih_full_ts.mean(dim=(1, 3))  # -> [B, D, D]

    # ------------------------------
    # 3b. Build lag dict in STI style
    # ------------------------------
    ih_lag_dict: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    # For each anchor timestep t and lag τ ≥ 0 (so t2 = t + τ)
    # and feature pair (d, d2), collect interactions across batch.
    # Shape per entry: [B].
    with torch.no_grad():
        for t in range(T):
            for tau in range(T):  # non-negative lags
                t2 = t + tau
                if t2 >= T:
                    break
                for d in range(D):
                    for d2 in range(D):
                        # vals: [B]
                        vals = ih_full_ts[:, t, d, t2, d2]  # [B]
                        key = (t, d, d2, tau)
                        ih_lag_dict[key] = vals.cpu().numpy()

    return ih_feat_matrix, ih_full_ts, ih_lag_dict


# ------------------------------
# Same aggregation we already use
# ------------------------------
def ih_main(
    model: torch.nn.Module,
    x: torch.Tensor,                   # [B, T, D]
    baseline: torch.Tensor,            # [1, T, D] or [B, T, D]
    m_steps: int = 50,
    target_idx: Optional[int] = 0,
):
    """
    Drop-in compatible with your existing aggregation snippet.


    Returns:
        lag_dict_mean   : dict[(tau, d, d2)] -> np.ndarray[B]
        lag_dict_median : dict[(tau, d, d2)] -> np.ndarray[B]
    """

    ih_feat_matrix, ih_full_ts, ih_lag_dict = integrated_hessians_timeseries(
        model,
        x=x,
        baseline=baseline,
        m_steps=m_steps,
        target_idx=target_idx,   # or whatever class/logit you care about
    )
    interactions = ih_lag_dict

    # Aggregate across anchor timesteps t
    agg: Dict[Tuple[int, int, int], list] = {}
    for (t, d, d2, tau), arr in interactions.items():  # arr: [B]
        agg.setdefault((tau, d, d2), []).append(arr)

    lag_dict_mean: Dict[Tuple[int, int, int], np.ndarray] = {
        (tau, d, d2): np.mean(np.stack(vals, axis=0), axis=0)  # [B]
        for (tau, d, d2), vals in agg.items()
    }
    lag_dict_median: Dict[Tuple[int, int, int], np.ndarray] = {
        (tau, d, d2): np.median(np.stack(vals, axis=0), axis=0)  # [B]
        for (tau, d, d2), vals in agg.items()
    }

    return lag_dict_mean, lag_dict_median
