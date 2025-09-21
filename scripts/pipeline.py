# scripts/pipeline.py
import argparse
import json
import os
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch as th
import pandas as pd

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, make_outdir
from src.models.tcn import TCNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.utils import make_loader, train_classifier
from src.explainers.sti import shapley_taylor_pairwise
from src.metrics.locality import aggregate_lag_curve, fit_decay, half_range, loc_at_k
from src.metrics.spectral import dft_magnitude, spectral_bandwidth, spectral_centroid, spectral_flatness
from src.utils.plotting import plot_locality, plot_spectrum


# ---------------------------------
# Sweep expansion (backward compatible)
# ---------------------------------
def expand_cfg(cfg):
    """Expand config into a list of configs if any param is a list; else [cfg]."""
    sweep_keys, sweep_vals = [], []
    for section in ["dataset", "model", "training", "experiment"]:
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            if isinstance(v, list):
                sweep_keys.append((section, k))
                sweep_vals.append(v)

    # Allow optional 'sweeps' block with key paths like 'model.name'
    if "sweeps" in cfg and isinstance(cfg["sweeps"], dict):
        for _, pairs in cfg["sweeps"].items():
            for keypath, values in pairs.items():
                if isinstance(values, list):
                    # keypath like "model.name" or "dataset.noise"
                    sect, key = keypath.split(".") if "." in keypath else ("experiment", keypath)
                    sweep_keys.append((sect, key))
                    sweep_vals.append(values)

    if not sweep_keys:
        return [cfg]

    configs = []
    for combo in product(*sweep_vals):
        new_cfg = {sec: dict(cfg[sec]) for sec in cfg if isinstance(cfg[sec], dict)}
        for (sec, k), val in zip(sweep_keys, combo):
            if sec not in new_cfg:
                new_cfg[sec] = {}
            new_cfg[sec][k] = val
        # carry over flags & sweeps unchanged
        for key in ["data_gen", "train", "compute_metrics", "sweeps"]:
            if key in cfg and key not in new_cfg:
                new_cfg[key] = cfg[key]
        configs.append(new_cfg)
    return configs


# ---------------------------------
# Dataset loaders
# ---------------------------------
def load_dataset(ds_cfg):
    """Generate a synthetic dataset based on config; returns (X, y, A, ds_name)."""
    name = ds_cfg["name"].lower()
    params = {k: v for k, v in ds_cfg.items() if k != "name"}
    if name == "var":
        from src.datasets.var import generate_var
        X, y, A = generate_var(**params)
    elif name == "arfima":
        from src.datasets.arfima import generate_arfima
        X, y = generate_arfima(**params)
        A = None
    elif name == "lorenz":
        from src.datasets.lorenz import generate_lorenz
        X, y = generate_lorenz(**params)
        A = None
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, A, name


def save_train_val_pickles(X, y, ds_name, split_ratio=0.8):
    n = len(X)
    split = int(split_ratio * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    data_dir = Path("data") / ds_name
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.pkl", "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)
    with open(data_dir / "val.pkl", "wb") as f:
        pickle.dump({"X": X_val, "y": y_val}, f)
    return (X_train, y_train), (X_val, y_val), data_dir


def load_pickles_if_exist(ds_name):
    data_dir = Path("data") / ds_name
    train_p = data_dir / "train.pkl"
    val_p = data_dir / "val.pkl"
    if train_p.exists() and val_p.exists():
        with open(train_p, "rb") as f:
            tr = pickle.load(f)
        with open(val_p, "rb") as f:
            va = pickle.load(f)
        return (tr["X"], tr["y"]), (va["X"], va["y"]), data_dir
    return None, None, data_dir


def class_stats(y):
    uniq, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


# ---------------------------------
# Training
# ---------------------------------
def select_model(cfg_model, D, C):
    name = str(cfg_model["name"]).lower()
    if name == "tcn":
        return TCNClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)), layers=int(cfg_model.get("layers", 2)))
    elif name == "lstm":
        return LSTMClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)))
    else:
        return TransformerClassifier(D, C, d_model=int(cfg_model.get("d_model", 64)))


def run_training(cfg, base_outdir):
    out = make_outdir(base_outdir, cfg, nested=True)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer existing pickles; otherwise generate
    ds_name = cfg["dataset"]["name"].lower()
    (train, val, data_dir) = load_pickles_if_exist(ds_name)
    if train is None:
        X, y, A, ds_name = load_dataset(cfg["dataset"])
        (X_train, y_train), (X_val, y_val), data_dir = save_train_val_pickles(X, y, ds_name)
    else:
        X_train, y_train = train
        X_val, y_val = val

    print(f"Dataset: {ds_name}")
    print(f"Train size: {len(y_train)} | Class balance: {class_stats(y_train)}")
    print(f"Val size:   {len(y_val)}   | Class balance: {class_stats(y_val)}")

    D = X_train.shape[2]
    C = len(np.unique(np.concatenate([y_train, y_val])))

    model = select_model(cfg["model"], D, C)

    train_loader = make_loader(X_train, y_train, batch=cfg["training"]["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, batch=cfg["training"]["batch_size"], shuffle=False)

    device = "cuda" if th.cuda.is_available() else "cpu"
    model, history = train_classifier(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=int(cfg["training"]["epochs"]),
        lr=1e-3,
        device=device,
        task_name=ds_name,
    )

    th.save(model.state_dict(), out / "model.pt")
    with open(out / "meta.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training complete. Saved to {out}")


# ---------------------------------
# Metrics (STI + locality/spectral)
# ---------------------------------
def compute_and_cache_lag_dicts(model, X_tensor, neighborhoods, tau_max, K, baseline, out_dir, device):
    """Load cached lag dicts if present; else compute and save."""
    lag_path_mean = out_dir / "lag_dict_mean.pkl"
    lag_path_median = out_dir / "lag_dict_median.pkl"

    if lag_path_mean.exists() and lag_path_median.exists():
        with open(lag_path_mean, "rb") as f:
            lag_dict_mean = pickle.load(f)
        with open(lag_path_median, "rb") as f:
            lag_dict_median = pickle.load(f)
        print(f"📂 Loaded cached lag_dicts from {out_dir}")
        return lag_dict_mean, lag_dict_median

    lag_dict_mean, lag_dict_median = shapley_taylor_pairwise(
        model=model,
        X=X_tensor,               # [N, T, D]
        tau_max=tau_max,
        neighborhoods=neighborhoods,
        K=K,
        baseline=baseline,
        cond_imputer=None,
        device=device,
    )
    with open(lag_path_mean, "wb") as f:
        pickle.dump(lag_dict_mean, f)
    with open(lag_path_median, "wb") as f:
        pickle.dump(lag_dict_median, f)
    print(f"✅ Computed and saved lag_dicts to {out_dir}")
    return lag_dict_mean, lag_dict_median


def fit_params_for_curve(curve):
    """Return (exp_fit, pow_fit) for a curve (accepts 1D or 2D)."""
    arr = np.array(curve)
    if arr.ndim == 1:
        arr = arr[:, None]   # make it (T,1) so fit_decay can handle shape[1]
    exp_p, pow_p = fit_decay(arr)
    exp_p = tuple(float(x) for x in np.array(exp_p).ravel()[:2])
    pow_p = tuple(float(x) for x in np.array(pow_p).ravel()[:2])
    return exp_p, pow_p



def maybe_stack_curves(curve):
    """
    Ensure we have both an aggregate 1D curve and (optionally) a 2D stack for 'all' plots.
    If input is dict->aggregate (1D), return (agg, None).
    If already 2D [tau+1, N], return (mean over N, full).
    """
    if isinstance(curve, np.ndarray) and curve.ndim == 2:
        agg = curve.mean(axis=1)
        return agg, curve
    return curve, None


def run_metrics(cfg, base_outdir):
    out = make_outdir(base_outdir, cfg, nested=True)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = out / "model.pt"
    if not ckpt_path.exists():
        print(f"❌ No model checkpoint found at {ckpt_path}, skipping metrics.")
        return

    # Prefer loading saved train data
    ds_name = cfg["dataset"]["name"].lower()
    (train, val, data_dir) = load_pickles_if_exist(ds_name)
    if train is None:
        # Fallback: regen data (keeps behavior consistent if pickles missing)
        X, y, A, _ = load_dataset(cfg["dataset"])
        X_train, y_train = X, y
    else:
        X_train, y_train = train

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Build model structure
    D = X_train.shape[2]
    C = len(np.unique(y_train))
    model = select_model(cfg["model"], D, C)
    model.load_state_dict(th.load(ckpt_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Use up to batch_size samples for STI (to limit compute), reproducible slice
    Bcap = int(cfg["training"]["batch_size"])
    X_t = th.tensor(X_train[:Bcap], dtype=th.float32, device=device)  # [N,T,D]

    neighborhoods = {d: [d] for d in range(D)}
    lag_mean, lag_median = compute_and_cache_lag_dicts(
        model=model,
        X_tensor=X_t,
        neighborhoods=neighborhoods,
        tau_max=int(cfg["experiment"]["tau_max"]),
        K=int(cfg["experiment"]["num_permutations"]),
        baseline="mean",
        out_dir=out,
        device=device,
    )

    # Curves (aggregate per tau from lag dicts)
    curve1 = aggregate_lag_curve(lag_mean, tau_max=int(cfg['experiment']['tau_max']))   # may be 1D or 2D if you changed impl
    curve2 = aggregate_lag_curve(lag_median, tau_max=int(cfg['experiment']['tau_max']))

    
    # Prepare aggregate + (optional) stacks for plotting 'all'
    agg1, stack1 = maybe_stack_curves(curve1)
    agg2, stack2 = maybe_stack_curves(curve2)

    # Fits
    exp_p1, pow_p1 = fit_params_for_curve(stack1)
    exp_p2, pow_p2 = fit_params_for_curve(stack2)

    # Plots
    # If stacks exist, plot mean ± std with faint traces (mode='all'); else just 1D
    plot_locality(stack1 if stack1 is not None else agg1,
                  out / "locality1.png",
                  fits={"exp": exp_p1, "pow": pow_p1},
                  mode="all" if stack1 is not None else "mean")
    plot_locality(stack2 if stack2 is not None else agg2,
                  out / "locality2.png",
                  fits={"exp": exp_p2, "pow": pow_p2},
                  mode="all" if stack2 is not None else "mean")

    # Spectral metrics on aggregate
    mag1 = dft_magnitude(agg1)
    B1 = spectral_bandwidth(mag1, 0.95)
    Cent1 = spectral_centroid(mag1)
    Flat1 = spectral_flatness(mag1)

    # Locality metrics on aggregate
    hr1 = half_range(agg1)
    lock1 = loc_at_k(stack1, K=min(10, len(stack1) - 1))

    # Save arrays
    np.save(out / "locality_curve1.npy", np.array(stack1))
    np.save(out / "locality_curve2.npy", np.array(stack1))
    np.save(out / "lock1.npy", np.array(lock1))
    plot_spectrum(mag1, out / "spectrum1.png")

    # Optionally save per-sample fit params if stacks exist
    if stack1 is not None:
        per_exp = []
        per_pow = []
        for j in range(stack1.shape[1]):
            e, p = fit_params_for_curve(stack1[:, j])
            per_exp.append(e)
            per_pow.append(p)
        np.save(out / "exp_fits_per_sample.npy", np.array(per_exp))  # [N,2]
        np.save(out / "pow_fits_per_sample.npy", np.array(per_pow))  # [N,2]

    # Summary (aggregate)
    summary = {
        "exp_fit": {"a": float(exp_p1[0]), "b": float(exp_p1[1])},
        "power_fit": {"a": float(pow_p1[0]), "p": float(pow_p1[1])},
        "half_range": int(hr1),
        # "loc_at_10": float(lock1),
        "bandwidth95": int(B1),
        "spec_centroid": float(Cent1),
        "spec_flatness": float(Flat1),
    }
    with open(out / "metrics1.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved metrics to {out / 'metrics1.json'}")


# ---------------------------------
# Data generation section (only if data_gen flag)
# ---------------------------------
def run_data_generation(cfg):
    ds_cfg = cfg["dataset"]
    X, y, A, ds_name = load_dataset(ds_cfg)
    (train, val, data_dir) = save_train_val_pickles(X, y, ds_name)
    X_train, y_train = train
    X_val, y_val = val
    print(f"📦 Data generated & saved to {data_dir}")
    print(f"Train size: {len(y_train)} | Class balance: {class_stats(y_train)}")
    print(f"Val size:   {len(y_val)}   | Class balance: {class_stats(y_val)}")


# ---------------------------------
# Aggregation over all sweeps
# ---------------------------------
def aggregate_all_metrics(base_outdir):
    outdir = Path(base_outdir)
    metrics_files = list(outdir.rglob("metrics*.json"))
    rows = []
    for mf in metrics_files:
        try:
            with open(mf, "r") as f:
                metrics = json.load(f)
            exp_id = str(mf.parent.relative_to(outdir))
            rows.append({"exp_id": exp_id, **metrics})
        except Exception as e:
            print(f"⚠️ Skipping {mf}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = outdir / "all_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Aggregated metrics written to {csv_path}")
    else:
        print("⚠️ No metrics files found to aggregate.")


# ---------------------------------
# Main
# ---------------------------------
def main(cfg_path, base_outdir):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    # Flags (backward compatible defaults)
    # - data_gen defaults to False (explicit request)
    # - train defaults to True (keeps previous behavior)
    # - compute_metrics defaults to True
    data_gen_flag = bool(cfg.get("data_gen", False))
    train_flag = bool(cfg.get("train", True))
    metrics_flag = bool(cfg.get("compute_metrics", True))

    for this_cfg in all_cfgs:
        if data_gen_flag:
            print("=== [DATA GEN] ===")
            run_data_generation(this_cfg)

        if train_flag:
            print("=== [TRAIN] ===")
            run_training(this_cfg, base_outdir)

        if metrics_flag:
            print("=== [METRICS] ===")
            run_metrics(this_cfg, base_outdir)

    # Aggregate across all runs (only if metrics were requested)
    if metrics_flag:
        aggregate_all_metrics(base_outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    args = ap.parse_args()
    main(args.config, args.base_outdir)
