import pickle
import numpy as np

from src.explainers.sti import shapley_taylor_pairwise
from src.explainers.ih import ih_main   
def get_interaction_curves(
    interaction_method, 
    model,
    X_tensor,
    neighborhoods,
    tau_max,
    K,
    baseline,
    device,
):
    """
    Compute or load full interaction curves.

    Returns:
        interaction_curves: np.ndarray of shape [N, T, tau_max+1, D, D]
            where:
                N  = batch size (X_tensor.shape[0])
                T  = time length (X_tensor.shape[1])
                D  = #features (X_tensor.shape[2])
                tau in {0, 1, ..., tau_max}
    """


    # ---- Core interaction computation ----
    # IMPORTANT: you need shapley_taylor_pairwise to return the *raw* per-t, per-tau dict.
    # i.e., interactions: {(t, d, d2, tau): np.ndarray[N]}
    # If your current version returns lag_dict_mean/median, refactor it so
    # there is an internal function that gives you this "interactions" dict,
    # and call that here.

    if (interaction_method == "sti"):

        interactions = shapley_taylor_pairwise(
            model=model,
            X=X_tensor,
            tau_max=tau_max,
            neighborhoods=neighborhoods,
            K=K,
            baseline=baseline,
            cond_imputer=None,
            device=device,
            # return_raw=True,  # <-- you'll add this flag in shapley_taylor_pairwise
        )
    elif (interaction_method == "ih"):
        interactions = ih_main(
            model=model,
            x=X_tensor,
            # tau_max=tau_max,
            # neighborhoods=neighborhoods,
            # K=K,
            baseline=baseline,
            # device=device,
            # return_raw=True,  # <-- you'll add this flag in ih_main
        )
    else:
        raise ValueError(f"Unknown interaction_method '{interaction_method}'")
    # --------------------------------------

    N, T, D = X_tensor.shape
    interaction_curves = np.zeros((N, T, tau_max + 1, D, D), dtype=np.float32)

    # interactions[(tau, d1, d2)] -> list of length (T - tau)
    # each element: arr_t: [N] for anchor time t
    for (tau, d1, d2), arr_list in interactions.items():
        if tau > tau_max:
            continue

        # arr_list[t] is defined only for t in [0, T - tau - 1]
        for t, arr in enumerate(arr_list):
            # arr: shape [N]
            interaction_curves[:, t, tau, d1, d2] = np.asarray(arr)

    return interaction_curves



def aggregate_curve(curves: np.ndarray, axis: str, mode: str = "mean") -> np.ndarray:
    """
    Generic aggregation over an axis of the interaction_curves tensor.

    curves shape:
        [N, T, tau_max+1, D, D]

    axis:
        'N'   -> aggregate over samples
        'T'   -> aggregate over anchor timesteps
        'tau' -> aggregate over lags
        'd1'  -> aggregate over source feature index
        'd2'  -> aggregate over target feature index

    mode:
        'mean' or 'median'

    Returns:
        reduced array with that axis removed.
    """
    axis_map = {"N": 0, "T": 1, "tau": 2, "d1": 3, "d2": 4}
    if axis not in axis_map:
        raise ValueError(f"Unknown axis '{axis}', expected one of {list(axis_map.keys())}")

    ax = axis_map[axis]
    if mode == "mean":
        return curves.mean(axis=ax)
    elif mode == "median":
        return np.median(curves, axis=ax)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'mean' or 'median'")
