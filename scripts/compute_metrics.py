import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch as th
import sys
from itertools import product

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, make_outdir
from src.metrics.locality import aggregate_lag_curve, fit_decay, half_range, loc_at_k
from src.metrics.spectral import dft_magnitude, spectral_bandwidth, spectral_centroid, spectral_flatness
from src.utils.plotting import plot_locality, plot_spectrum
from src.explainers.sti import shapley_taylor_pairwise
from src.models.tcn import TCNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


# -------------------------------
# Sweep handling
# -------------------------------
def expand_cfg(cfg):
    """Expand config into list of configs if any param is a list."""
    sweep_keys, sweep_vals = [], []
    for section in ["dataset", "model", "training", "experiment"]:
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            if isinstance(v, list):
                sweep_keys.append((section, k))
                sweep_vals.append(v)

    if not sweep_keys:
        return [cfg]  # single run

    configs = []
    for combo in product(*sweep_vals):
        new_cfg = {sec: dict(cfg[sec]) for sec in cfg}
        for (sec, k), val in zip(sweep_keys, combo):
            new_cfg[sec][k] = val
        configs.append(new_cfg)
    return configs


# -------------------------------
# Core metric computation
# -------------------------------
def run_metrics(cfg, base_outdir):
    # derive experiment folder (same rule as train_model.py)
    out = make_outdir(base_outdir, cfg, nested=True)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = out / "model.pt"
    if not ckpt_path.exists():
        print(f"❌ No model checkpoint found at {ckpt_path}, skipping.")
        return

    # --- Load synthetic data (regenerate for now) ---
    from src.datasets.var import generate_var
    X, y, A = generate_var(**cfg['dataset'])
    X_t = th.tensor(X, dtype=th.float32)[:cfg["training"]["batch_size"]]

    D = X.shape[2]
    C = 2
    model_name = cfg['model']['name'].lower()
    if model_name == 'tcn':
        model = TCNClassifier(D, C, hidden=64, layers=2)
    elif model_name == 'lstm':
        model = LSTMClassifier(D, C, hidden=64)
    else:
        model = TransformerClassifier(D, C, d_model=64)

    model.load_state_dict(th.load(ckpt_path, map_location="cpu"))
    model.eval()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # --- Load or compute lag dicts ---
    lag_path_mean = out / "lag_dict_mean.pkl"
    lag_path_median = out / "lag_dict_median.pkl"
    if lag_path_mean.exists() and lag_path_median.exists():
        with open(lag_path_mean, "rb") as f:
            lag_dict_mean = pickle.load(f)
        with open(lag_path_median, "rb") as f:
            lag_dict_median = pickle.load(f)
        print(f"📂 Loaded cached lag_dicts from {out}")
    else:
        neighborhoods = {d: [d] for d in range(D)}
        lag_dict_mean, lag_dict_median = shapley_taylor_pairwise(
            model,
            X_t,
            tau_max=cfg['experiment']['tau_max'],
            neighborhoods=neighborhoods,
            K=cfg['experiment']['num_permutations'],
            baseline="mean",
            cond_imputer=None,
            device=device,
        )
        with open(lag_path_mean, "wb") as f:
            pickle.dump(lag_dict_mean, f)
        with open(lag_path_median, "wb") as f:
            pickle.dump(lag_dict_median, f)
        print(f"✅ Computed and saved lag_dicts to {out}")

    # --- Curves ---
    curve1 = aggregate_lag_curve(lag_dict_mean, tau_max=cfg['experiment']['tau_max'])
    curve2 = aggregate_lag_curve(lag_dict_median, tau_max=cfg['experiment']['tau_max'])

    # --- Fits and plots ---
    exp_p1, pow_p1 = fit_decay(curve1)
    exp_p2, pow_p2 = fit_decay(curve2)
    a, b = exp_p1.mean(axis=0)
    c, p = pow_p1.mean(axis=0)
    plot_locality(curve1, out / "locality1.png", fits={"exp": exp_p1, "pow": pow_p1}, mode='all')
    plot_locality(curve2, out / "locality2.png", fits={"exp": exp_p2, "pow": pow_p2}, mode='all')

    # --- Metrics ---
    hr1 = half_range(curve1)
    lock1 = loc_at_k(curve1, K=min(10, len(curve1)-1))
    mag1 = dft_magnitude(curve1)
    B1 = spectral_bandwidth(mag1, 0.95)
    Cent1 = spectral_centroid(mag1)
    Flat1 = spectral_flatness(mag1)

    # Save
    np.save(out / "locality_curve1.npy", curve1)
    np.save(out / "locality_curve2.npy", curve2)

    summary = {
        "exp_fit": {"a": float(a), "b": float(b)},
        "power_fit": {"a": float(c), "p": float(p)},
        "half_range": int(hr1),
        "loc_at_10": float(lock1),
        "bandwidth95": int(B1),
        "spec_centroid": float(Cent1),
        "spec_flatness": float(Flat1),
    }
    with open(out / "metrics1.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved metrics to {out / 'metrics1.json'}")


def to_scalar(x, idx=None):
    arr = np.array(x).ravel()   # flatten everything
    if idx is not None:
        return float(arr[idx])
    if arr.size == 1:
        return float(arr[0])
    raise ValueError(f"Expected scalar, got array of size {arr.size}")



# -------------------------------
# Main
# -------------------------------
def main(cfg_path, base_outdir):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    for this_cfg in all_cfgs:
        run_metrics(this_cfg, base_outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    args = ap.parse_args()
    main(args.config, args.base_outdir)
