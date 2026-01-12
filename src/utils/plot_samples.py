# src/utils/plot_samples.py
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


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




def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def plot_explainer_samples(
    X,
    y,
    attributions,
    expl_name,
    expl_dir,
    num_samples: int = 5,
):
    """
    Save visualizations for a few samples under expl_dir/plots/.

    Parameters
    ----------
    X : array-like (N, T, D)
    y : array-like (N,)
    attributions : array-like (N, T, D)
    expl_name : str
    expl_dir : str
    num_samples : int
    """

    X = _to_numpy(X)
    y = _to_numpy(y)
    attributions = _to_numpy(attributions)

    plots_dir = os.path.join(expl_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    N, T, D = X.shape
    num_samples = min(num_samples, N)

    # evenly spaced samples
    sample_ids = np.linspace(0, N - 1, num_samples, dtype=int)

    # shared attribution color scale (important!)
    vmax = np.percentile(np.abs(attributions), 99)
    vmin = -vmax

    for idx, sid in enumerate(sample_ids):

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 7),
            dpi=180,
            sharex=True,
        )

        ax_signal, ax_label, ax_attr = axes

        # ------------------ 1) Raw Signal ------------------
        ax_signal.imshow(
            X[sid].T,
            aspect="auto",
            cmap="Greys",
            interpolation="nearest",
        )
        ax_signal.set_title(f"Signal (sample {sid})")
        ax_signal.set_ylabel("Features")
        ax_signal.grid(alpha=0.25)

        # ------------------ 2) Label ------------------
        label_row = np.ones((1, T)) * y[sid]
        ax_label.imshow(
            label_row,
            aspect="auto",
            cmap="Greens",
            vmin=0,
            vmax=1,
        )
        ax_label.set_yticks([])
        ax_label.set_title(f"Label = {y[sid]}")
        ax_label.grid(alpha=0.25)

        # ------------------ 3) Attribution ------------------
        im = ax_attr.imshow(
            attributions[sid].T,
            aspect="auto",
            cmap="RdBu_r",   # diverging = much better for attribution
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_attr.set_title(f"{expl_name} Attribution")
        ax_attr.set_ylabel("Features")
        ax_attr.set_xlabel("Time")
        ax_attr.grid(alpha=0.25)

        # colorbar
        cbar = plt.colorbar(im, ax=ax_attr, fraction=0.015, pad=0.02)
        cbar.set_label("Attribution")

        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"{expl_name}_sample_{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
