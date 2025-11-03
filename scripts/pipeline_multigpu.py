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
from src.metrics.locality import aggregate_lag_curve, fit_decay, half_range, loc_at_k, loc_at_50
from src.metrics.spectral import dft_magnitude, spectral_bandwidth, spectral_centroid, spectral_flatness
from src.utils.plotting import plot_locality, plot_spectrum

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------
# Pretty printing
# ---------------------------------
def banner(msg: str):
    print("\n" + "=" * 80)
    print(f"🔹 {msg}")
    print("=" * 80 + "\n")


# ---------------------------------
# Sweep expansion
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

    if "sweeps" in cfg and isinstance(cfg["sweeps"], dict):
        for _, pairs in cfg["sweeps"].items():
            for keypath, values in pairs.items():
                if isinstance(values, list):
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
        for key in ["data_gen", "train", "compute_metrics", "sweeps"]:
            if key in cfg and key not in new_cfg:
                new_cfg[key] = cfg[key]
        configs.append(new_cfg)
    return configs


# ---------------------------------
# Dataset utils
# ---------------------------------
def load_dataset(ds_cfg):
    name = ds_cfg["name"].lower()
    params = {k: v for k, v in ds_cfg.items() if k != "name"}
    if name == "var":
        from src.datasets.var import generate_var
        X, y, A = generate_var(**params)

    if name == "var_local":
        from src.datasets.var_planted import generate_var
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
    train_p, val_p = data_dir / "train.pkl", data_dir / "val.pkl"
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
    print(out)
    
    ckpt = out / "model.pt"
    if ckpt.exists():
        banner(f"Skipping training — already exists: {ckpt}")
        return

    out.mkdir(parents=True, exist_ok=True)
    ds_name = cfg["dataset"]["name"].lower()
    (train, val, _) = load_pickles_if_exist(ds_name)
    if train is None:
        X, y, A, ds_name = load_dataset(cfg["dataset"])
        (X_train, y_train), (X_val, y_val), _ = save_train_val_pickles(X, y, ds_name)
    else:
        X_train, y_train = train
        X_val, y_val = val

    print(f"Dataset: {ds_name}")
    print(f"Train size: {len(y_train)} | Class balance: {class_stats(y_train)}")
    print(f"Val size:   {len(y_val)}   | Class balance: {class_stats(y_val)}")

    D, C = X_train.shape[2], len(np.unique(np.concatenate([y_train, y_val])))
    model = select_model(cfg["model"], D, C)

    train_loader = make_loader(X_train, y_train, batch=cfg["training"]["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, batch=cfg["training"]["batch_size"], shuffle=False)

    device = "cuda" if th.cuda.is_available() else "cpu"
    model, history = train_classifier(
        model, train_loader, val_loader=val_loader,
        epochs=int(cfg["training"]["epochs"]), lr=1e-3,
        device=device, task_name=ds_name,
    )

    th.save(model.state_dict(), ckpt)
    with open(out / "meta.json", "w") as f: json.dump(cfg, f, indent=2)
    with open(out / "history.json", "w") as f: json.dump(history, f, indent=2)
    print(f"✅ Training complete. Saved to {out}")


# ---------------------------------
# Metrics
# ---------------------------------
def compute_and_cache_lag_dicts(model, X_tensor, neighborhoods, tau_max, K, baseline, out_dir, device):
    lag_path_mean, lag_path_median = out_dir / "lag_dict_mean.pkl", out_dir / "lag_dict_median.pkl"
    if lag_path_mean.exists() and lag_path_median.exists():
        with open(lag_path_mean, "rb") as f: lag_dict_mean = pickle.load(f)
        with open(lag_path_median, "rb") as f: lag_dict_median = pickle.load(f)
        print(f"📂 Loaded cached lag_dicts from {out_dir}")
        return lag_dict_mean, lag_dict_median

    lag_dict_mean, lag_dict_median = shapley_taylor_pairwise(
        model=model, X=X_tensor, tau_max=tau_max,
        neighborhoods=neighborhoods, K=K, baseline=baseline,
        cond_imputer=None, device=device,
    )
    with open(lag_path_mean, "wb") as f: pickle.dump(lag_dict_mean, f)
    with open(lag_path_median, "wb") as f: pickle.dump(lag_dict_median, f)
    print(f"✅ Computed and saved lag_dicts to {out_dir}")
    return lag_dict_mean, lag_dict_median

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
    # print(out)
    metrics_file = out / "metrics1.json"
    # if metrics_file.exists():
    #     banner(f"Skipping metrics — already exists: {metrics_file}")
    #     return

    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "model.pt"
    if not ckpt_path.exists():
        print(f"❌ No checkpoint at {ckpt_path}, skipping metrics.")
        return

    ds_name = cfg["dataset"]["name"].lower()
    (train, _, _) = load_pickles_if_exist(ds_name)
    if train is None:
        X, y, A, _ = load_dataset(cfg["dataset"])
        X_train = X
    else:
        X_train, _ = train

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    D, C = X_train.shape[2], len(np.unique(_))
    model = select_model(cfg["model"], D, C)
    model.load_state_dict(th.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()

    Bcap = int(cfg["training"]["batch_size"])
    X_t = th.tensor(X_train[:Bcap], dtype=th.float32, device=device)

    neighborhoods = {d: [d] for d in range(D)}
    lag_mean, lag_median = compute_and_cache_lag_dicts(
        model, X_t, neighborhoods,
        tau_max=int(cfg["experiment"]["tau_max"]),
        K=int(cfg["experiment"]["num_permutations"]),
        baseline="mean", out_dir=out, device=device,
    )

    curves1 = aggregate_lag_curve(lag_mean, 
                                  tau_max=int(cfg['experiment']['tau_max']),
                                   reduce="mean")
    
    
    K = min(cfg["evals"]["loc@k"], len(curves1)-1)
    locK = loc_at_k(curves1, K)
    loc50 = loc_at_50(curves1)

    print(f"Loc@{K}: {locK:.4f}")
    print(f"Loc@50: {loc50}")

    agg1, curves1 = maybe_stack_curves(curves1)

    exp_p1, pow_p1 = fit_decay(np.array(curves1))
    exp_mean = exp_p1.mean(axis=0)   # shape (2,)
    pow_mean = pow_p1.mean(axis=0)   # shape (2,)

    mag1 = dft_magnitude(agg1)
    summary = {
        "exp_fit": {"a": float(exp_mean[0]), "b": float(exp_mean[1])},
        "power_fit": {"a": float(pow_mean[0]), "p": float(pow_mean[1])},
        "half_range": int(half_range(agg1)),
        "loc_at_k": locK,
        "loc50": loc50,
        "bandwidth95": int(spectral_bandwidth(mag1, 0.95)),
        "spec_centroid": float(spectral_centroid(mag1)),
        "spec_flatness": float(spectral_flatness(mag1)),
    }
    with open(metrics_file, "w") as f: json.dump(summary, f, indent=2)
    print(f"✅ Metrics saved to {metrics_file}")


# ---------------------------------
# Aggregation
# ---------------------------------
def aggregate_all_metrics(base_outdir):
    outdir = Path(base_outdir)
    metrics_files = list(outdir.rglob("metrics*.json"))
    rows = []
    for mf in metrics_files:
        try:
            with open(mf, "r") as f: metrics = json.load(f)
            exp_id = str(mf.parent.relative_to(outdir))
            rows.append({"exp_id": exp_id, **metrics})
        except Exception as e:
            print(f"⚠️ Skipping {mf}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        tsv_path = outdir / "all_metrics.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        banner(f"📊 Aggregated metrics written to {tsv_path}")
    else:
        banner("⚠️ No metrics files found to aggregate.")


# ------------------------------
# Multi-GPU scheduling
# ------------------------------
def _run_one(i, total, this_cfg, base_outdir, gpu_id, flags):
    """
    Worker that runs one expanded config on a single GPU (or CPU if gpu_id is None).
    We pin the process with CUDA_VISIBLE_DEVICES so downstream code can just use 'cuda'.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        which = f"GPU {gpu_id}"
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        which = "CPU"

    banner(f"[{which}] Sweep {i+1}/{total}")

    if flags["data_gen"]:
        X, y, A, ds_name = load_dataset(this_cfg["dataset"])
        save_train_val_pickles(X, y, ds_name)
        print(f"📦 Data generated for {ds_name}")

    if flags["train"]:
        run_training(this_cfg, base_outdir)

    if flags["metrics"]:
        run_metrics(this_cfg, base_outdir)

    return i

def main(cfg_path, base_outdir, gpus_arg="0,1,2,3", max_workers=None):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    flags = {
        "data_gen": bool(cfg.get("data_gen", False)),
        "train": bool(cfg.get("train", True)),
        "metrics": bool(cfg.get("compute_metrics", True)),
    }

    # Parse list of GPUs (e.g., "0,1,2,3"); allow empty to force CPU
    gpus_arg = (gpus_arg or "").strip()
    gpu_ids = [int(x) for x in gpus_arg.split(",") if x != ""] if gpus_arg else []

    if max_workers is None:
        max_workers = max(1, len(gpu_ids)) if gpu_ids else mp.cpu_count()

    # Round-robin assignment of configs to GPUs
    assignments = []
    for i, this_cfg in enumerate(all_cfgs):
        gpu = None if not gpu_ids else gpu_ids[i % len(gpu_ids)]
        assignments.append((i, this_cfg, gpu))

    # Use 'spawn' to be CUDA-safe in subprocesses
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = []
        for i, this_cfg, gpu in assignments:
            futures.append(ex.submit(_run_one, i, len(all_cfgs), this_cfg, base_outdir, gpu, flags))
        # Raise early if any worker fails
        for f in as_completed(futures):
            _ = f.result()

    if flags["metrics"]:
        aggregate_all_metrics(base_outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated physical GPU IDs (e.g., '0,1,2,3'). Empty string for CPU.")
    ap.add_argument("--max_workers", type=int, default=None,
                    help="#processes. Default: one per GPU; CPU=logical cores.")
    args = ap.parse_args()
    main(args.config, args.base_outdir, gpus_arg=args.gpus, max_workers=args.max_workers)

