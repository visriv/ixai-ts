import numpy as np
import json
from pathlib import Path

def prf1(y_true, y_pred, eps=1e-12):
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def auc_roc(y_true, scores):
    """Rank-based AUROC (no sklearn)."""
    y_true = y_true.astype(int)
    scores = scores.astype(float)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Mann–Whitney U
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = ranks[:len(pos)]

    U = r_pos.sum() - len(pos) * (len(pos) - 1) / 2
    return float(U / (len(pos) * len(neg)))


def auc_pr(y_true, scores):
    """Simple PR-AUC via sorted threshold sweep."""
    order = np.argsort(-scores)
    y_true = y_true[order]

    tp = 0
    fp = 0
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return float("nan")

    precisions = []
    recalls = []

    for y in y_true:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)

    return float(np.trapz(precisions, recalls))





def compute_pairwise_metrics(
    I_model,     # [tau, D, D]
    A_gt,        # [L, D, D]  (lags 1..L)
    cfg,
    out_dir: Path,
):
    """
    Computes Section 3.7 metrics:
      (a) Graph recovery (lag-sensitive & lag-agnostic) + AUROC/AUPRC
      (b) Interaction–lag locality
      (c) Strength calibration (Spearman)
    """

    eps = 1e-12
    tau_max, D, _ = I_model.shape
    delta = float(cfg["evals"].get("delta", 0.0))
    K_loc = int(cfg["evals"].get("loc@k", 1))

    # --------------------------------------------------
    # Build GT interaction tensor I_true[tau,i,j]
    # --------------------------------------------------
    I_true = np.zeros_like(I_model)
    L = min(len(A_gt), tau_max - 1)
    for k in range(L):
        I_true[k + 1] = np.abs(A_gt[k])

    # ==================================================
    # (a) GRAPH RECOVERY
    # ==================================================

    # ---------- Lag-sensitive (triples) ----------
    mask = np.ones_like(I_true, dtype=bool)
    mask[0] = False                          # ignore tau=0
    mask[:, np.arange(D), np.arange(D)] = False

    y_true_tri = (I_true > 0)[mask]
    scores_tri = np.abs(I_model)[mask]
    y_pred_tri = scores_tri >= delta

    prf_tri = prf1(y_true_tri, y_pred_tri)
    auroc_tri = auc_roc(y_true_tri, scores_tri)
    auprc_tri = auc_pr(y_true_tri, scores_tri)

    # ---------- Lag-agnostic (pairs) ----------
    I_true_pair = np.sum(I_true, axis=0)
    I_model_pair = np.sum(np.abs(I_model), axis=0)

    mask2 = ~np.eye(D, dtype=bool)
    y_true_pair = (I_true_pair > 0)[mask2]
    scores_pair = I_model_pair[mask2]
    y_pred_pair = scores_pair >= delta

    prf_pair = prf1(y_true_pair, y_pred_pair)
    auroc_pair = auc_roc(y_true_pair, scores_pair)
    auprc_pair = auc_pr(y_true_pair, scores_pair)

    # ==================================================
    # (b) INTERACTION–LAG LOCALITY
    # ==================================================
    lag_errors = []
    locKs = []

    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            true_lags = np.where(I_true[:, i, j] > 0)[0]
            if len(true_lags) == 0:
                continue

            tau_star = true_lags[0]
            curve = np.abs(I_model[:, i, j])
            curve[0] = 0.0

            Z = curve.sum()
            if Z < eps:
                continue

            p = curve / Z
            lag_errors.append(np.sum(p * np.abs(np.arange(tau_max) - tau_star)))
            locKs.append(np.sum(p[np.abs(np.arange(tau_max) - tau_star) <= K_loc]))

    lag_error = float(np.mean(lag_errors)) if lag_errors else float("nan")
    locK_mean = float(np.mean(locKs)) if locKs else float("nan")

    # ==================================================
    # (c) STRENGTH CALIBRATION
    # ==================================================
    true_s = []
    pred_s = []

    for tau in range(1, tau_max):
        for i in range(D):
            for j in range(D):
                if i != j and I_true[tau, i, j] > 0:
                    true_s.append(I_true[tau, i, j])
                    pred_s.append(abs(I_model[tau, i, j]))

    if len(true_s) >= 2:
        r_true = np.argsort(np.argsort(true_s))
        r_pred = np.argsort(np.argsort(pred_s))
        spearman = float(np.corrcoef(r_true, r_pred)[0, 1])
    else:
        spearman = float("nan")

    # ==================================================
    # SAVE
    # ==================================================
    results = {
        "graph_recovery": {
            "lag_sensitive": {
                **prf_tri,
                "auroc": auroc_tri,
                "auprc": auprc_tri,
            },
            "lag_agnostic": {
                **prf_pair,
                "auroc": auroc_pair,
                "auprc": auprc_pair,
            },
            "threshold_delta": delta,
        },
        "lag_locality": {
            "mean_lag_error": lag_error,
            f"mean_loc@{K_loc}": locK_mean,
        },
        "strength_calibration": {
            "spearman": spearman,
        },
    }

    out_path = out_dir / "pairwise_faithfulness_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved pairwise faithfulness metrics to {out_path}")
    return results
