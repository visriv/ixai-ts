# src/utils/plot_samples.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_timeseries(
    X: np.ndarray,
    ds_name: str,
    sample_idx: int = 0,
    out_root: Path = Path("data"),
    fname: Optional[str] = None,
) -> Path:
    """
    Plot one multivariate time series sample as D stacked subplots.

    Parameters
    ----------
    X : np.ndarray
        Array of shape [N, T, D].
    ds_name : str
        Dataset name (used for folder: data/{ds_name}/plots/).
    sample_idx : int, default 0
        Which sequence index to plot.
    out_root : Path, default Path("data")
        Root directory for saving plots.
    fname : str, optional
        File name for the PNG. If None, use f"sample_{sample_idx}.png".

    Returns
    -------
    out_path : Path
        Full path of the saved PNG.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, T, D], got {X.shape}")

    N, T, D = X.shape
    if not (0 <= sample_idx < N):
        raise IndexError(f"sample_idx {sample_idx} out of range [0, {N-1}]")

    seq = X[sample_idx]        # [T, D]

    # output directory: data/{ds_name}/plots/
    out_dir = out_root / ds_name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if fname is None:
        fname = f"sample_{sample_idx}.png"
    out_path = out_dir / fname

    # Matplotlib style tweaks: serif, thin grid, readable labels
    plt.rcParams.update({
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(
        D, 1,
        figsize=(7, 1.8 * D),
        sharex=True,
        constrained_layout=True,
    )

    # If D == 1, axes is not a list
    if D == 1:
        axes = [axes]

    t = np.arange(T)

    y_min = float(seq.min())
    y_max = float(seq.max())
    margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= margin
    y_max += margin

    for d in range(D):
        ax = axes[d]
        ax.plot(t, seq[:, d], lw=2)
        ax.set_ylabel(f"Feature {d+1}", fontsize=11)
        # Only show x ticks on bottom subplot
        if d != D - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Time steps", fontsize=12)

    fig.suptitle(f"{ds_name} – sample {sample_idx}", fontsize=13)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path
