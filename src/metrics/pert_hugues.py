import torch as th
import numpy as np


def compute_hugues_metrics(curves):
    fracs = curves["fractions"]
    top = curves["hug_curve_top"]
    bot = curves["hug_curve_bottom"]

    hug_auc_top = float(np.trapz(top, fracs))
    hug_auc_bottom = float(np.trapz(bot, fracs))

    hug_f1 = (
        hug_auc_top * (1 - hug_auc_bottom)
        / (hug_auc_top + (1 - hug_auc_bottom) + 1e-12)
    )

    return {
        "hug_auc_top": hug_auc_top,
        "hug_auc_bottom": hug_auc_bottom,
        "hug_f1": hug_f1,
    }

@th.no_grad()
def compute_hughes_curves(
    model,
    X,
    y,
    attributions,
    cfg):
    """
    Computes Hugues-style top/bottom corruption curves.

    Returns:
      dict with keys:
        hug_curve_top
        hug_curve_bottom
        fractions
        f_x
    """

    model.eval()
    device = X.device
    N, T, D = X.shape
    TD = T * D

    fractions = cfg["pointwise"]["evals"]["k_list"]  # e.g. [0.01,0.05,0.1,0.2]
    baseline = cfg["pointwise"]["evals"]["baseline"]

    # ---- rank features ----
    # flat_attr = th.from_numpy(attributions).abs().view(N, -1)
    flat_attr = th.from_numpy(attributions).reshape(N, -1)


    rank_desc = th.argsort(flat_attr, dim=1, descending=True)
    rank_asc = th.argsort(flat_attr, dim=1, descending=False)

    # ---- original score ----
    out = model(X)
    out = out[:, -1, :] if out.dim() == 3 else out
    f_x = out.gather(1, y.view(-1, 1)).mean().item()

    def corrupt(X, rank, k):
        Xc = X.clone()
        for n in range(N):
            flat = Xc[n].view(-1)
            idx = rank[n, :k]
            if baseline == "mean":
                flat[idx] = flat.mean()
            else:
                flat[idx] = 0.0
        return Xc

    hug_top, hug_bottom = [], []

    for frac in fractions:
        k = max(1, int(frac * TD))

        # ---- top-k corruption ----
        Xt = corrupt(X, rank_desc, k)
        out = model(Xt)
        out = out[:, -1, :] if out.dim() == 3 else out
        f_t = out.gather(1, y.view(-1, 1)).mean().item()
        hug_top.append((f_x - f_t) / (f_x + 1e-12))

        # ---- bottom-k corruption ----
        Xb = corrupt(X, rank_asc, k)
        out = model(Xb)
        out = out[:, -1, :] if out.dim() == 3 else out
        f_b = out.gather(1, y.view(-1, 1)).mean().item()
        hug_bottom.append((f_x - f_b) / (f_x + 1e-12))

    return {
        "fractions": np.array(fractions),
        "hug_curve_top": np.array(hug_top),
        "hug_curve_bottom": np.array(hug_bottom),
        "f_x": f_x,
    }
