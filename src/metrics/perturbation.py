import torch as th
import numpy as np


def rank_features(attr):
    # attr: [N,T,D] or [N,D]
    if not isinstance(attr, th.Tensor):
        attr = th.from_numpy(attr)

    flat = attr.reshape(attr.shape[0], -1)
    # flat = attr.abs().reshape(attr.shape[0], -1)
    return th.argsort(flat, dim=1, descending=True)


def apply_deletion(X, rank_idx, k, baseline="zero"):
    Xp = X.clone()
    for n in range(X.shape[0]):
        flat = Xp[n].view(-1)
        if baseline == "mean":
            flat[rank_idx[n, :k]] = flat.mean()
        else:
            flat[rank_idx[n, :k]] = 0.0
    return Xp


def apply_insertion(X, rank_idx, k, baseline="zero"):
    if baseline == "mean":
        Xp = X.mean(dim=1, keepdim=True).expand_as(X).clone()
    else:
        Xp = th.zeros_like(X)

    for n in range(X.shape[0]):
        flat_src = X[n].view(-1)
        flat_dst = Xp[n].view(-1)
        flat_dst[rank_idx[n, :k]] = flat_src[rank_idx[n, :k]]
    return Xp

def compute_pointwise_metrics_from_curves(curves):
    f_x = curves["f_x"]
    f_base = curves["f_base"]

    del_curve = curves["deletion"]
    ins_curve = curves["insertion"]
    suff_curve = curves["sufficiency"]

    aopc_del = np.mean(f_x - del_curve)
    aopc_ins = np.mean(ins_curve - f_base)
    suff = np.mean(f_x - suff_curve)
    comp = np.mean(f_x - del_curve)

    return {
        "AOPC_deletion": float(aopc_del),
        "AOPC_insertion": float(aopc_ins),
        "Sufficiency": float(suff),
        "Comprehensiveness": float(comp),
    }

    

@th.no_grad()
def compute_perturbation_curves(model, X, y, attr, cfg):
    model.eval()

    N, T, D = X.shape
    total_feats = T * D

    fractions = cfg["pointwise"]["evals"]["k_list"]  # e.g. [0.01,0.05,0.1,0.2]
    baseline = cfg["pointwise"]["evals"]["baseline"]

    rank_idx = rank_features(attr)

    # original score
    out0 = model(X)
    out0 = out0[:, -1, :] if out0.dim() == 3 else out0
    f_x = out0.gather(1, y.view(-1,1)).mean().item()

    # baseline score
    X_base = th.zeros_like(X)
    out_base = model(X_base)
    out_base = out_base[:, -1, :] if out_base.dim() == 3 else out_base
    f_base = out_base.gather(1, y.view(-1,1)).mean().item()

    del_scores, ins_scores, suff_scores = [], [], []

    for frac in fractions:
        k = max(1, int(frac * total_feats))

        # deletion
        X_del = apply_deletion(X, rank_idx, k, baseline)
        out = model(X_del)
        out = out[:, -1, :] if out.dim() == 3 else out
        del_scores.append(out.gather(1, y.view(-1,1)).mean().item())

        # insertion
        X_ins = apply_insertion(X, rank_idx, k, baseline)
        out = model(X_ins)
        out = out[:, -1, :] if out.dim() == 3 else out
        ins_scores.append(out.gather(1, y.view(-1,1)).mean().item())

        # sufficiency (keep only top-k)
        X_suff = apply_deletion(X, rank_idx, total_feats - k, baseline)
        out = model(X_suff)
        out = out[:, -1, :] if out.dim() == 3 else out
        suff_scores.append(out.gather(1, y.view(-1,1)).mean().item())

    return {
        "fractions": fractions,
        "f_x": f_x,
        "f_base": f_base,
        "deletion": np.array(del_scores),
        "insertion": np.array(ins_scores),
        "sufficiency": np.array(suff_scores),
    }
