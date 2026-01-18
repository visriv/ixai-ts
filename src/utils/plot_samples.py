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
    feature_idx: int = 0,
):

    X = _to_numpy(X)
    y = _to_numpy(y)
    attributions = _to_numpy(attributions)

    plots_dir = os.path.join(expl_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    N, T, D = X.shape
    num_samples = min(num_samples, N)
    sample_ids = np.linspace(0, N - 1, num_samples, dtype=int)

    vmax = np.percentile(np.abs(attributions), 99)
    vmin = -vmax

    for idx, sid in enumerate(sample_ids):

        fig, axes = plt.subplots(
            4, 1,
            figsize=(14, 9),
            dpi=180,
            sharex=True,
        )

        ax_signal, ax_label, ax_attr, ax_overlay = axes

        # ------------------ 1) Raw Signal (all features) ------------------
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

        # ------------------ 3) Attribution (all features) ------------------
        im = ax_attr.imshow(
            attributions[sid].T,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_attr.set_title(f"{expl_name} Attribution")
        ax_attr.set_ylabel("Features")
        ax_attr.grid(alpha=0.25)

        cbar = plt.colorbar(im, ax=ax_attr, fraction=0.015, pad=0.02)
        cbar.set_label("Attribution")

        # ------------------ 4) OVERLAY: feature_0 attribution + signal ------------------
        attr_f = attributions[sid, :, feature_idx]   # (T,)
        signal_f = X[sid, :, feature_idx]             # (T,)

        # Attribution as background (1 x T heatmap)
        ax_overlay.imshow(
            attr_f[None, :],
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Overlay raw signal
        ax_overlay.plot(
            np.arange(T),
            signal_f,
            color="black",
            linewidth=2.0,
        )

        ax_overlay.set_yticks([])
        ax_overlay.set_ylabel(f"f{feature_idx}")
        ax_overlay.set_title(
            f"Feature {feature_idx}: raw signal over attribution"
        )
        ax_overlay.set_xlabel("Time")
        ax_overlay.grid(alpha=0.25)

        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"{expl_name}_sample_{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()


def plot_explainer_samples_old(
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




def plot_insertion_deletion_curves(
    curves: dict,
    expl_name: str,
    expl_dir: str,
):
    """
    Plot insertion & deletion perturbation curves.

    curves keys expected:
      - fractions: list[float]
      - deletion: np.ndarray
      - insertion: np.ndarray
      - f_x: float (original score)
      - f_base: float (baseline score)
    """

    expl_dir = Path(expl_dir)
    expl_dir.mkdir(parents=True, exist_ok=True)

    fracs = np.array(curves["fractions"])
    del_curve = np.array(curves["deletion"])
    ins_curve = np.array(curves["insertion"])

    f_x = curves.get("f_x", None)
    f_base = curves.get("f_base", None)

    # -----------------------------
    # Matplotlib style (global)
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # -----------------------------
    # Deletion curve
    # -----------------------------
    ax.plot(
        fracs, del_curve,
        label="Deletion",
        color="#d62728",        # muted red
        linewidth=2.5,
        zorder=3,
    )
    ax.fill_between(
        fracs, del_curve,
        del_curve.min(),
        color="#d62728",
        alpha=0.18,
        zorder=2,
    )

    # -----------------------------
    # Insertion curve
    # -----------------------------
    ax.plot(
        fracs, ins_curve,
        label="Insertion",
        color="#1f77b4",        # muted blue
        linewidth=2.5,
        zorder=3,
    )
    ax.fill_between(
        fracs, ins_curve,
        ins_curve.min(),
        color="#1f77b4",
        alpha=0.18,
        zorder=2,
    )

    # -----------------------------
    # Reference lines
    # -----------------------------
    if f_x is not None:
        ax.axhline(
            f_x,
            linestyle="--",
            color="black",
            linewidth=1.2,
            alpha=0.7,
            label="Original score",
        )

    if f_base is not None:
        ax.axhline(
            f_base,
            linestyle=":",
            color="black",
            linewidth=1.2,
            alpha=0.6,
            label="Baseline score",
        )

    # -----------------------------
    # Labels & cosmetics
    # -----------------------------
    ax.set_xlabel("Fraction of features perturbed")
    ax.set_ylabel("Model score")
    ax.set_title(f"Insertion / Deletion Curves — {expl_name}")

    ax.set_xlim(fracs.min(), fracs.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    out_path = expl_dir / "plots" / f"{expl_name}_insertion_deletion.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Saved insertion/deletion plot to {out_path}")


import matplotlib.pyplot as plt


def plot_hughes_curves(curves, expl_name, expl_dir):
    fracs = curves["fractions"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
    })

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    ax.plot(fracs, curves["hug_curve_top"],
            label="Top-k corrupted",
            linewidth=2.5)
    ax.fill_between(fracs, curves["hug_curve_top"], alpha=0.25)

    ax.plot(fracs, curves["hug_curve_bottom"],
            label="Bottom-k corrupted",
            linewidth=2.5)
    ax.fill_between(fracs, curves["hug_curve_bottom"], alpha=0.25)

    ax.set_xlabel("Fraction of points removed")
    ax.set_ylabel("Normalized score drop")
    ax.set_title(f"Faithfulness — {expl_name}")
    ax.legend(frameon=False)

    out = f"{expl_dir}/plots/{expl_name}_hugues_curve.pdf"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"📈 Saved Hughes curve to {out}")
