import numpy as np
from scipy.stats import spearmanr

def interaction_alignment_score(attributions, interactions, topk=None):
    # attributions: 1D array (flattened relevance per cell)
    # interactions: dict {(u,v): I_uv}
    mu, sigma = attributions.mean(), attributions.std()
    norm_attr = (attributions - mu) / (sigma + 1e-8)

    saliency_pairs = {}
    for (u,v), I_uv in interactions.items():
        saliency_pairs[(u,v)] = norm_attr[u] * norm_attr[v]

    pairs = list(interactions.keys())
    pred = [saliency_pairs[p] for p in pairs]
    true = [interactions[p] for p in pairs]

    if topk is not None:
        idx = np.argsort(np.abs(true))[-topk:]
        pred = np.array(pred)[idx]
        true = np.array(true)[idx]

    corr, _ = spearmanr(pred, true)
    return corr
