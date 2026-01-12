# src/metrics/perturbation.py

import torch as th
import numpy as np

def _rank_features(attr):
    """
    attr: torch.Tensor or np.ndarray
          shape [N,T,D] or [N,D]
    returns:
        rank_idx: torch.LongTensor [N, TD]
    """
    if not isinstance(attr, th.Tensor):
        attr = th.from_numpy(attr)

    flat = attr.abs().view(attr.shape[0], -1)
    return th.argsort(flat, dim=1, descending=True)



def _mask_k(X, rank_idx, k, baseline="zero"):
    Xp = X.clone()
    idx = rank_idx[:, :k]

    for n in range(X.shape[0]):
        flat = Xp[n].view(-1)
        if baseline == "mean":
            flat[idx[n]] = flat.mean()
        else:
            flat[idx[n]] = 0.0
    return Xp


@th.no_grad()
def compute_deletion_curve(model, X, y, attr, cfg):
    k_max = cfg["pointwise"]["perturbation"]["k_max"]
    baseline = cfg["pointwise"]["perturbation"]["baseline"]

    rank_idx = _rank_features(attr)
    scores = []

    for k in range(1, k_max + 1):
        Xk = _mask_k(X, rank_idx, k, baseline)
        out = model(Xk)
        out = out[:, -1, :] if out.dim() == 3 else out
        score = out.gather(1, y.view(-1, 1)).mean().item()
        scores.append(score)

    return np.array(scores)


@th.no_grad()
def compute_insertion_curve(model, X, y, attr, cfg):
    baseline = cfg["pointwise"]["perturbation"]["baseline"]
    k_max = cfg["pointwise"]["perturbation"]["k_max"]

    rank_idx = _rank_features(attr)
    X0 = th.zeros_like(X)
    scores = []

    for k in range(1, k_max + 1):
        Xk = _mask_k(X, rank_idx, k, baseline=None)
        out = model(Xk)
        out = out[:, -1, :] if out.dim() == 3 else out
        score = out.gather(1, y.view(-1, 1)).mean().item()
        scores.append(score)

    return np.array(scores)


def compute_AOPC(ins_curve, del_curve):
    """
    Area Over Perturbation Curve
    """
    return float(np.mean(del_curve) - np.mean(ins_curve))
