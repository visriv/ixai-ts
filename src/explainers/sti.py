import math
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
      cond_imputer: optional module g(X, S_keep[B,T,D]) -> X_filled for 'conditional'
    Returns:
      Xm: [N, T, D]
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)  # -> [1, T, D]
    N, T, D = X.shape

    keep = S_keep.bool().unsqueeze(0).expand(N, -1, -1)  # [N,T,D]

    if baseline == "conditional" and cond_imputer is not None:
        Xm = cond_imputer(X, keep)  # user-provided imputer
    else:
        if baseline == "mean":
            # per-sample, per-feature mean across time
            base = X.mean(dim=1, keepdim=True)  # [N,1,D]
        else:
            base = th.zeros_like(X)
        Xm = th.where(keep, X, base)
    return Xm


def value_function(
    model: nn.Module,
    X: th.Tensor,
    S_keep: th.Tensor,
    baseline: str = "zero",
    cond_imputer: Optional[nn.Module] = None,
    return_logits: bool = False,
    device: Optional[str] = None,
) -> th.Tensor:
    """
    v(S): per-sample KL divergence KL(p || q_S)
    Args:
      X: [N,T,D] or [T,D]
      S_keep: [T,D] boolean
    Returns:
      kl: [N]  (per-sample values; NO averaging over N)
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    model = model.to(device)   # <-- ADD THIS LINE

    if X.dim() == 2:
        Xb = X.unsqueeze(0).to(device)
    else:
        Xb = X.to(device)

    with th.no_grad():
        p = model(Xb)                # [N,C]
        p = th.softmax(p, dim=-1)    # [N,C]

    Xm = mask_input(Xb, S_keep.to(device), baseline, cond_imputer)  # [N,T,D]
    q = model(Xm)                    # [N,C]
    q = th.softmax(q, dim=-1)

    # KL per sample: sum over classes only
    kl = (p * (p.add(1e-9).log() - q.add(1e-9).log())).sum(dim=-1)  # [N]
    return kl  # keep per-sample, no mean()


def random_permutation(U_size: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(U_size)
    rng.shuffle(perm)
    return perm


def coords_to_index(t: int, d: int, T: int, D: int) -> int:
    return t * D + d


def index_to_coords(idx: int, T: int, D: int) -> Tuple[int, int]:
    return idx // D, idx % D


@th.no_grad()
def shapley_taylor_pairwise(
    model: nn.Module,
    X: th.Tensor,                                   # [N,T,D] or [T,D]
    tau_max: int,
    neighborhoods: Dict[int, Iterable[int]],        # feature d -> iterable of d'
    K: int = 100,                                   # permutations per pair
    baseline: str = "zero",
    cond_imputer: Optional[nn.Module] = None,
    importance_map: Optional[np.ndarray] = None,    # (unused placeholder)
    device: Optional[str] = None,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Returns:
      lag_dict: {(tau, d, d'): np.ndarray[N]}  # per-sample interactions
    Computes pairwise STI for ((t,d),(t+tau,d')) with 0<=tau<=tau_max, d'∈neighborhoods[d].
    For each (u,v), we estimate:
        I^(n)(u,v) = median_{k=1..K} [ v^(n)(T_k ∪ {u,v}) - v^(n)(T_k ∪ {u})
                                       - v^(n)(T_k ∪ {v}) + v^(n)(T_k) ]
    Then aggregate across valid t by median (per-sample preserved).
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    # Ensure batch dimension present
    if X.dim() == 2:
        X = X.unsqueeze(0)  # [1,T,D]
    X = X.to(device)

    N, T, D = X.shape
    U = T * D
    rng = np.random.default_rng(seed=0)

    def coalition_from_pred(pred_set: set) -> th.Tensor:
        mask = th.zeros((T, D), dtype=th.bool, device=device)
        for idx in pred_set:
            t, d = index_to_coords(idx, T, D)
            mask[t, d] = True
        return mask  # [T,D]

    # (t, d, d', tau) -> np.ndarray[N]
    interactions = {}

    for t in tqdm(range(T), desc="STI lags", leave=True):
        max_tau = min(tau_max, T - 1 - t)
        for tau in tqdm(range(max_tau + 1), desc=f"tau loop (t={t})", leave=False):
            t2 = t + tau
            for d in range(D):
                neigh = neighborhoods.get(d, [d])
                for d2 in neigh:
                    u = coords_to_index(t,  d,  T, D)
                    v = coords_to_index(t2, d2, T, D)
                    if u == v:
                        continue

                    divs = []  # list of arrays [N]
                    for _ in range(K):
                        # Predecessor set via permutation positions
                        perm = random_permutation(U, rng)
                        pos_u = int(np.where(perm == u)[0][0])
                        pos_v = int(np.where(perm == v)[0][0])
                        cutoff = min(pos_u, pos_v)
                        Tset = set(perm[:cutoff])

                        S_T   = coalition_from_pred(Tset)
                        S_Tu  = coalition_from_pred(Tset | {u})
                        S_Tv  = coalition_from_pred(Tset | {v})
                        S_Tuv = coalition_from_pred(Tset | {u, v})

                        v_T   = value_function(model, X, S_T,   baseline, cond_imputer, device=device)   # [N]
                        v_Tu  = value_function(model, X, S_Tu,  baseline, cond_imputer, device=device)   # [N]
                        v_Tv  = value_function(model, X, S_Tv,  baseline, cond_imputer, device=device)   # [N]
                        v_Tuv = value_function(model, X, S_Tuv, baseline, cond_imputer, device=device)   # [N]

                        diffs = (v_Tuv - v_Tu - v_Tv + v_T).cpu().numpy()  # [N]
                        divs.append(diffs)

                    # median over permutations, preserving [N]
                    print(np.mean(np.array(divs)))
                    divs = np.stack(divs, axis=0)  # [K,N]
                    interactions[(t, d, d2, tau)] = np.mean(divs, axis=0)  # [N]

    # Aggregate across anchor timesteps (median over t), keep [N]
    agg: Dict[Tuple[int, int, int], list] = {}
    for (t, d, d2, tau), arr in interactions.items():  # arr: [N]
        agg.setdefault((tau, d, d2), []).append(arr)

    lag_dict_mean: Dict[Tuple[int, int, int], np.ndarray] = {
        (tau, d, d2): np.mean(np.stack(vals, axis=0), axis=0)  # [N]
        for (tau, d, d2), vals in agg.items()
    }
    lag_dict_median: Dict[Tuple[int, int, int], np.ndarray] = {
        (tau, d, d2): np.median(np.stack(vals, axis=0), axis=0)  # [N]
        for (tau, d, d2), vals in agg.items()
    }
    return lag_dict_mean, lag_dict_median
