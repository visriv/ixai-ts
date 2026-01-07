# src/datasets/ih_mult.py

from typing import Dict, Tuple, Optional

import numpy as np


def _g_multiplicative(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Base interaction function g(x1, x2) = x1 * x2."""
    return x1 * x2


def generate_ih_multi(
    num_samples: int = 400,
    num_features: int = 5,
    num_interactions: int = 1,
    noise: float = 0.05,
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Multiplicative synthetic regression task from the Integrated Hessians paper.

    Data:
        X_n ~ N(0, 1) independently, shape [N, D].

    Ground-truth label:
        y_n = sum_{i=1}^M alpha_i * g(x_{n, i1}, x_{n, i2}) + eps_n

    where:
        - pairs (i1, i2) are drawn without replacement from all unordered feature pairs
        - alpha_i ~ Uniform(0, 1), normalised so sum_i alpha_i = 1
        - g(x1, x2) = x1 * x2 (multiplicative interaction)
        - eps_n ~ N(0, noise_std^2) is optional label noise

    Parameters
    ----------
    num_samples : int
        Number of data points N.
    num_features : int
        Number of features D.
    num_interactions : int
        Number of interacting pairs M to plant (must be <= D choose 2).
    noise_std : float
        Standard deviation of additive Gaussian label noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape [N, D]
        Feature matrix.
    y : np.ndarray, shape [N]
        Regression labels.
    meta : dict
        Ground-truth interaction info:
            - "pairs": [M, 2] array of feature indices (i1, i2)
            - "alpha": [M] array of interaction weights summing to 1
            - "A": [D, D] symmetric matrix with interaction strength per pair
    """
    rng = np.random.default_rng(seed)

    # X ~ N(0, 1)  [N, D]
    X = rng.normal(loc=0.0, scale=1.0, size=(num_samples, num_features))

    # All unordered feature pairs (i < j)
    all_pairs = [(i, j) for i in range(num_features) for j in range(i + 1, num_features)]
    num_all_pairs = len(all_pairs)
    if num_interactions > num_all_pairs:
        raise ValueError(
            f"num_interactions={num_interactions} exceeds number of possible "
            f"pairs C(D,2)={num_all_pairs} for D={num_features}"
        )


    '''
    multiple interactions
    '''
    # # Sample interaction pairs without replacement
    # chosen_idx = rng.choice(num_all_pairs, size=num_interactions, replace=False)
    # pairs = np.array([all_pairs[k] for k in chosen_idx], dtype=int)  # [M, 2]
    # # Compute interaction contributions for each sample
    # pair_contributions[n, i] = g(x_{n, i1}, x_{n, i2})
    # pair_contributions = np.empty((num_samples, num_interactions), dtype=float)
    # for i, (p1, p2) in enumerate(pairs):
    #     pair_contributions[:, i] = _g_multiplicative(X[:, p1], X[:, p2])

    # # Interaction weights alpha_i ~ U(0,1), then normalize to sum to 1
    # alpha_raw = rng.uniform(0.0, 1.0, size=num_interactions)
    # alpha = alpha_raw / alpha_raw.sum()

    # y = np.sum(X, axis=-1) + (pair_contributions * alpha[None, :]).sum(axis=1)
    
    # for w, (p1, p2) in zip(alpha, pairs):
    #     A[p1, p2] += w
    #     A[p2, p1] += w   # symmetric
    '''
    single interaction
    '''
    alpha = 2
    y = np.sum(X, axis=-1) + alpha * np.prod(X[:, 0:2], axis=-1)
    
    # Optional additive Gaussian noise on labels
    if noise > 0.0:
        y += rng.normal(loc=0.0, scale=noise, size=num_samples)

    # Ground-truth interaction matrix A[d, d]:
    A = np.zeros((num_features, num_features), dtype=float)


    A[0,1] = alpha
    A[1,0] = alpha

    X_all = X.reshape((num_samples, 1, num_features)) # NTD
    y_all = y # (N,) 
    A_list = [A] # wrap A in a list to match VAR format, len(A_list) = k = 1
    return X_all, y_all, A_list
