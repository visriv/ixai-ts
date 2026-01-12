import numpy as np
import torch as th
from torch import nn
from typing import Dict, Tuple, Iterable, Optional
from tqdm import tqdm


def mask_input(
    X: th.Tensor,
    S_keep: th.Tensor,
    baseline: str = "zero",
    cond_imputer: Optional[nn.Module] = None,
) -> th.Tensor:
    """
    Masking utility.
    Args:
      X: [N, T, D] or [T, D]    (if 2D, will be unsqueezed to batch=1)
      S_keep: [T, D] boolean, True where we KEEP the value
      baseline: 'zero' | 'mean' | 'conditional'
      cond_imputer: optional module g(X, keep_mask[N,T,D]) -> X_filled for 'conditional'
    Returns:
      Xm: [N, T, D]
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)  # -> [1, T, D]
    N, T, D = X.shape

    keep = S_keep.bool().unsqueeze(0).expand(N, -1, -1)  # [N,T,D]

    if baseline == "conditional" and cond_imputer is not None:
        Xm = cond_imputer(X, keep)
    else:
        if baseline == "mean":
            base = X.mean(dim=1, keepdim=True)  # [N,1,D]
        else:
            base = th.zeros_like(X)
        Xm = th.where(keep, X, base)
    return Xm


@th.no_grad()
def value_function(
    model: nn.Module,
    X: th.Tensor,
    S_keep: th.Tensor,
    target: Optional[th.Tensor] = None,
    target_idx: Optional[int] = None,
    baseline: str = "zero",
    cond_imputer: Optional[nn.Module] = None,
    output: str = "logit",
    device: Optional[str] = None,
) -> th.Tensor:
    """
    Game value v(S) for STI/SII: scalar model output on masked input.

    Args:
      model: classifier/regressor
      X: [N,T,D] or [T,D]
      S_keep: [T,D] boolean mask (True = keep real value)
      target_idx: fixed class index k (int) to score (classification)
      target: per-sample target class indices [N] (classification)
      output: 'logit' | 'prob' | 'regression'
        - 'logit': returns logits[..., k]
        - 'prob' : returns softmax prob[..., k]
        - 'regression': returns model(Xm).squeeze(-1)

    Returns:
      v: [N] tensor
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    if X.dim() == 2:
        Xb = X.unsqueeze(0).to(device)
    else:
        Xb = X.to(device)

    Xm = mask_input(Xb, S_keep.to(device), baseline, cond_imputer)  # [N,T,D]
    out = model(Xm)

    # Handle per-time outputs: if [N,T,C], take last time step by default
    if out.dim() == 3:
        out = out[:, -1, :]  # [N,C]

    if output == "regression":
        return out.squeeze(-1)  # [N] (or [N,] after squeeze)

    # Classification paths below
    if out.dim() != 2:
        raise ValueError(f"Expected classifier output [N,C] (or [N,T,C]), got shape {tuple(out.shape)}")

    if target is None:
        if target_idx is None:
            # Default: use model's predicted class per sample (stable, but note it's mask-dependent)
            # If you prefer "always the original prediction on unmasked X", pass target explicitly.
            target = out.argmax(dim=-1)
        else:
            target = th.full((out.shape[0],), int(target_idx), device=out.device, dtype=th.long)
    else:
        target = target.to(out.device).long()

    if output == "logit":
        return out.gather(1, target.view(-1, 1)).squeeze(1)  # [N]

    if output == "prob":
        probs = th.softmax(out, dim=-1)
        return probs.gather(1, target.view(-1, 1)).squeeze(1)  # [N]

    raise ValueError(f"Unknown output='{output}'. Use 'logit'|'prob'|'regression'.")


def coords_to_index(t: int, d: int, D: int) -> int:
    return t * D + d


def index_to_coords(idx: int, D: int) -> Tuple[int, int]:
    return idx // D, idx % D


def _coalition_mask_from_indices(indices: np.ndarray, T: int, D: int, device: str) -> th.Tensor:
    """
    Build [T,D] boolean mask where indices are flattened positions in [0..T*D-1].
    """
    mask = th.zeros((T, D), dtype=th.bool, device=device)
    if indices.size == 0:
        return mask
    # Vectorized scatter
    idx_t = th.from_numpy(indices // D).to(device)
    idx_d = th.from_numpy(indices % D).to(device)
    mask[idx_t, idx_d] = True
    return mask


@th.no_grad()
def shapley_taylor_pairwise(
    model: nn.Module,
    X: th.Tensor,                                   # [N,T,D] or [T,D]
    tau_max: int,
    neighborhoods: Dict[int, Iterable[int]],        # feature d -> iterable of d'
    K: int = 100,                                   # permutations per pair
    baseline: str = "zero",
    cond_imputer: Optional[nn.Module] = None,
    target: Optional[th.Tensor] = None,             # [N] class indices (recommended)
    target_idx: Optional[int] = None,               # fixed class index
    output: str = "logit",                          # 'logit' | 'prob' | 'regression'
    agg_over_t: str = "mean",                       # 'mean' | 'median'
    device: Optional[str] = None,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Pairwise Shapley–Taylor Index (order 2) for time-series coordinates.

    We treat each time-feature coordinate (t,d) as a "player".
    For pair u=(t,d), v=(t+tau, d2), STI(2) is approximated by:
        E_{π}[ Δ_{uv} v( Pre_π({u,v}) ) ]
    where Pre_π({u,v}) are players before BOTH u and v (before the earlier one).

    Returns:
      out: {(tau, d, d2): np.ndarray[N]}  # per-sample interactions, aggregated over anchor t
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if X.dim() == 2:
        X = X.unsqueeze(0)
    X = X.to(device)

    N, T, D = X.shape
    U = T * D
    rng = np.random.default_rng(seed=0)

    # If user didn't pass per-sample target labels for classification,
    # choose them ONCE from the unmasked input (recommended for stability).
    if output in ("logit", "prob") and target is None and target_idx is None:
        model = model.to(device)
        model.eval()
        with th.no_grad():
            out0 = model(X)
            if out0.dim() == 3:
                out0 = out0[:, -1, :]
            target = out0.argmax(dim=-1)  # [N]

    # Accumulate per (tau,d,d2) a list of [N] arrays across anchor times t
    bucket: Dict[Tuple[int, int, int], list] = {}

    for t in tqdm(range(T), desc="STI(2) anchors", leave=True):
        max_tau = min(tau_max, T - 1 - t)
        for tau in range(max_tau + 1):
            t2 = t + tau
            for d in range(D):
                neigh = list(neighborhoods.get(d, [d]))
                for d2 in neigh:
                    u = coords_to_index(t,  d,  D)
                    v = coords_to_index(t2, d2, D)
                    if u == v:
                        continue

                    # Collect K samples of Δ_uv v(Pre)
                    diffs_K = []

                    for _ in range(K):
                        perm = np.arange(U)
                        rng.shuffle(perm)

                        pos_u = int(np.where(perm == u)[0][0])
                        pos_v = int(np.where(perm == v)[0][0])
                        cutoff = min(pos_u, pos_v)  # predecessors before both (before earlier one)
                        pre = perm[:cutoff]         # numpy array

                        S_T   = _coalition_mask_from_indices(pre,             T, D, device)
                        S_Tu  = _coalition_mask_from_indices(np.append(pre, u), T, D, device)
                        S_Tv  = _coalition_mask_from_indices(np.append(pre, v), T, D, device)
                        S_Tuv = _coalition_mask_from_indices(np.append(pre, [u, v]), T, D, device)

                        v_T   = value_function(model, X, S_T,   target=target, target_idx=target_idx,
                                              baseline=baseline, cond_imputer=cond_imputer, output=output, device=device)
                        v_Tu  = value_function(model, X, S_Tu,  target=target, target_idx=target_idx,
                                              baseline=baseline, cond_imputer=cond_imputer, output=output, device=device)
                        v_Tv  = value_function(model, X, S_Tv,  target=target, target_idx=target_idx,
                                              baseline=baseline, cond_imputer=cond_imputer, output=output, device=device)
                        v_Tuv = value_function(model, X, S_Tuv, target=target, target_idx=target_idx,
                                              baseline=baseline, cond_imputer=cond_imputer, output=output, device=device)

                        diffs = (v_Tuv - v_Tu - v_Tv + v_T).detach().cpu().numpy()  # [N]
                        diffs_K.append(diffs)

                    diffs_K = np.stack(diffs_K, axis=0)  # [K,N]
                    sti_uv_at_t = diffs_K.mean(axis=0)   # [N]  (use median here if you want robustness)

                    bucket.setdefault((tau, d, d2), []).append(sti_uv_at_t)

    # Aggregate across anchor times t
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, arrs in bucket.items():
        A = np.stack(arrs, axis=0)  # [num_t, N]
        if agg_over_t == "median":
            out[key] = np.median(A, axis=0)
        else:
            out[key] = A.mean(axis=0)

    return out
