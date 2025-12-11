import torch
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from tqdm.auto import tqdm
# if torch.cuda.is_available():
#     torch.backends.cuda.sdp_kernel(
#         enable_flash=False,
#         enable_mem_efficient=False,
#         enable_math=True,
#     )

from torch.nn.attention import sdpa_kernel, SDPBackend
# ------------------------------
# 1. Scalar output helper
# ------------------------------
def _scalar_output(model: torch.nn.Module,
                   x: torch.Tensor,
                   target_idx: Optional[int] = None) -> torch.Tensor:
    """
    x: [B, T, D]
    Returns a scalar F(x) to attribute.
    """
    logits = model(x)   # e.g. [B, T, C]

    if logits.ndim == 3:
        # choose class and aggregate over time
        B, T, C = logits.shape
        if target_idx is None:
            # e.g. pick argmax class, then mean over T
            target_idx = logits.mean(dim=1).argmax(dim=1).item()
        # scalar: mean over time and batch for that class
        return logits[..., target_idx].mean()

    elif logits.ndim == 2:
        # old case: [B, C]
        if target_idx is None:
            target_idx = logits.argmax(dim=1).item()
        return logits[:, target_idx]#.mean()
    else:
        # already scalar
        return logits.squeeze()



# ------------------------------
# 2. Base IH on a flattened input
# ------------------------------
def integrated_hessians_flat(
    model: torch.nn.Module,
    x_mat: torch.Tensor,        # [T, D]
    baseline_mat: torch.Tensor, # [T, D]
    m_steps: int = 50,
    target_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Integrated Hessians for a single time-series example.

    We conceptually flatten x ∈ R^{T×D} into a vector of length F = T*D,
    but we always reshape back to [1, T, D] before calling the model.
    """
    model.eval()

    # Shapes
    T, D = x_mat.shape
    device = x_mat.device

    # Flatten to a 1-D coordinate vector for autograd
    x_vec = x_mat.detach().reshape(-1)          # [F]
    baseline_vec = baseline_mat.detach().reshape(-1)  # [F]
    delta = x_vec - baseline_vec                # [F]
    F = delta.shape[0]

    # f: R^F -> scalar, but internally calls model on [1, T, D]
    def f(z_flat: torch.Tensor) -> torch.Tensor:
        """
        z_flat: [F]
        """
        z_ts = z_flat.view(1, T, D)             # [1, T, D]
        return _scalar_output(model, z_ts, target_idx=target_idx)

    # Integration nodes in (0, 1]; skip α=0 for stability
    alphas = torch.linspace(0.0, 1.0, steps=m_steps + 1, device=device)[1:]

    H_sum = torch.zeros(F, F, device=device)

    with sdpa_kernel(SDPBackend.MATH):
        for alpha in alphas:
            x_alpha = baseline_vec + alpha * delta  # [F]
            H = torch.autograd.functional.hessian(f, x_alpha)  # [F, F]
            H_sum += H

    H_avg = H_sum / alphas.numel()  # approximate ∫_0^1 H(x' + αΔx) dα

    # Feature scaling: (Δx_i Δx_j)
    d = delta.view(-1)                              # [F]
    IH_flat = H_avg * (d.view(F, 1) * d.view(1, F)) # [F, F]

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

    for b in tqdm(range(B), desc="IH over batch"):
        x_b = x[b]          # [T, D]
        base_b = baseline[b]  # [T, D]

        IH_flat_b = integrated_hessians_flat(
            model=model,
            x_mat=x_b,
            baseline_mat=base_b,
            m_steps=m_steps,
            target_idx=target_idx,
        )  # [F, F] with F = T*D

        IH_ts_b = IH_flat_b.view(T, D, T, D)
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
    baseline: str,         # "mean", [1, T, D] or [B, T, D]
    m_steps: int = 50,
    target_idx: Optional[int] = 0,
):
    """
    Drop-in compatible with your existing aggregation snippet.


    Returns:
        lag_dict_mean   : dict[(tau, d, d2)] -> np.ndarray[B]
        lag_dict_median : dict[(tau, d, d2)] -> np.ndarray[B]
    """
    baseline_tensor = x.mean(dim=0, keepdim=True)  # [1, T, D]
    ih_feat_matrix, ih_full_ts, ih_lag_dict = integrated_hessians_timeseries(
        model,
        x=x,
        baseline=baseline_tensor,
        m_steps=m_steps,
        target_idx=target_idx,   # or whatever class/logit you care about
    )
    interactions = ih_lag_dict

    # Aggregate across anchor timesteps t
    agg: Dict[Tuple[int, int, int], list] = {}
    for (t, d, d2, tau), arr in interactions.items():  # arr: [B]
        agg.setdefault((tau, d, d2), []).append(arr)

    return agg#lag_dict_mean, lag_dict_median
