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

from src.utils.aggregating import get_interaction_curves, aggregate_curve

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.tcn import TCNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.utils import make_loader, train_classifier
from src.explainers.pairwise.sti import shapley_taylor_pairwise
from src.explainers.pairwise.ih import ih_main
from src.metrics.locality import aggregate_lag_curve, fit_decay, half_range, loc_at_k, loc_at_50
from src.metrics.spectral import dft_magnitude, spectral_bandwidth, spectral_centroid, spectral_flatness
from src.metrics.pairwise_metrics import compute_pairwise_metrics
from src.metrics.pointwise_metrics import compute_pointwise_metrics
from src.utils.plotting import plot_locality, plot_spectrum
from src.utils.plot_samples import plot_sample_timeseries
from src.utils.loading import load_pickles_if_exist, load_dataset
from src.utils.config import load_config, make_outdir

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
    for section in ["model", "training", "experiment"]:
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





def class_stats(y):
    uniq, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}







# ---------------------------------
# Training
# ---------------------------------
def select_model(cfg_model, D, C, all_times=False):
    name = str(cfg_model["name"]).lower()
    if name == "tcn":
        return TCNClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)), layers=int(cfg_model.get("layers", 2)))
    elif name == "lstm":
        return LSTMClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)))
    else:
        return TransformerClassifier(D, C, d_model=int(cfg_model.get("d_model", 64)), all_times=all_times)


def run_training(cfg, base_outdir):
    out = make_outdir(base_outdir, cfg, nested=True)
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

    D, _ = X_train.shape[2], len(np.unique(np.concatenate([y_train, y_val])))
    # if cfg["dataset"]["all_times"]:
        # cfg["dataset"]["multi_label"]:
        # pass
        # C = y_train.shape[1]
    # else:
    C = max(2, len(np.unique(y_train)))

    model = select_model(cfg["model"], D, C, cfg["dataset"]["all_times"])

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


def run_pairwise_xai(cfg, base_outdir):
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
    D = X_train.shape[2]
    C = max(2, len(np.unique(_)))
    model = select_model(cfg["model"], D, C, cfg["dataset"]["all_times"])
    model.load_state_dict(th.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()

    Bcap = int(cfg["training"]["batch_size"])
    X_t = th.tensor(X_train[:Bcap], dtype=th.float32, device=device)

    neighborhoods = {d: [d] for d in range(D)}
    N, T, D = X_t.shape

    interaction_method = cfg["experiment"]["interaction_method"]
    interaction_curves_path = out / f"interaction_curves_{interaction_method}.pkl"


    if interaction_curves_path.exists():
        with open(interaction_curves_path, "rb") as f:
            interaction_curves = pickle.load(f)
        print(f"📂 Loaded cached interaction_curves from {interaction_curves_path}")
    else:    
        interaction_curves = get_interaction_curves(
            interaction_method = cfg["experiment"]["interaction_method"],
            model=model,
            X_tensor=X_t,
            neighborhoods=neighborhoods,
            tau_max=int(cfg["experiment"]["tau_max"]),
            K=int(cfg["experiment"]["num_permutations"]),
            baseline="mean",

            device=device,
        )
        with open(interaction_curves_path    , "wb") as f:
            pickle.dump(interaction_curves, f)
        print(f"✅ Computed and saved interaction_curves to {interaction_curves_path}")


    # 1) aggregate over time T
    agg_T_curves = aggregate_curve(
        interaction_curves,
        axis="T",
        mode="mean",  # or "mean"
    )
    # save
    with open(out / "agg_T_interaction_curves.pkl", "wb") as f:
        pickle.dump(agg_T_curves, f)

    # 2) aggregate over tau
    agg_T_N_curves = aggregate_curve(
        agg_T_curves,
        axis="N",
        mode="mean",  # or "mean"
    )
    with open(out / "agg_T_N_interaction_curves.pkl", "wb") as f:
        pickle.dump(agg_T_N_curves, f)

    #  feature index # TODO
    curves1 = agg_T_N_curves[:,0,0]
    curves1_N = agg_T_curves[:,:,0,0]  # [tau, N]
    K = min(cfg["evals"]["loc@k"], T)
    locK = loc_at_k(curves1, K)
    loc50 = loc_at_50(curves1)

    print(f"Loc@{K}: {locK:.4f}")
    print(f"Loc@50: {loc50}")

    # agg1, curves1 = maybe_stack_curves(curves1)

    exp_p1, pow_p1 = fit_decay(np.array(curves1_N))
    exp_mean = exp_p1.mean(axis=0)   # shape (2,)
    pow_mean = pow_p1.mean(axis=0)   # shape (2,)

    mag1 = dft_magnitude(curves1)
    summary = {
        "exp_fit": {"a": float(exp_mean[0]), "b": float(exp_mean[1])},
        "power_fit": {"a": float(pow_mean[0]), "p": float(pow_mean[1])},
        "half_range": int(half_range(curves1)),
        "loc_at_k": locK,
        "loc50": loc50,
        "bandwidth95": int(spectral_bandwidth(mag1, 0.95)),
        "spec_centroid": float(spectral_centroid(mag1)),
        "spec_flatness": float(spectral_flatness(mag1)),
    }
    with open(metrics_file, "w") as f: json.dump(summary, f, indent=2)
    print(f"✅ Metrics saved to {metrics_file}")


    compute_pairwise_metrics(out, cfg)



def run_pointwise_xai(cfg, base_outdir):
    compute_pointwise_metrics(cfg, base_outdir)
    



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


# ---------------------------------
# Main
# ---------------------------------
def main(cfg_path, base_outdir):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    data_gen_flag = bool(cfg.get("data_gen", False))
    train_flag = bool(cfg.get("train", True))
    pointwise_xai_flag = bool(cfg.get("pointwise_xai", True))
    pairwise_xai_flag = bool(cfg.get("pairwise_xai", True))

    for i, this_cfg in enumerate(all_cfgs):
        banner(f"Running sweep {i+1}/{len(all_cfgs)} → {this_cfg}")
        if data_gen_flag:
            X, y, A, ds_name = load_dataset(this_cfg["dataset"])
            plot_sample_timeseries(X, ds_name, sample_idx=0)
            (X_train, y_train), (X_val, y_val), data_dir = \
                save_train_val_pickles(X, y, ds_name)
            print(f"📦 Data generated for {ds_name}")
        if train_flag:
            run_training(this_cfg, base_outdir)
        if pointwise_xai_flag:
            run_pointwise_xai(this_cfg, base_outdir)
        if pairwise_xai_flag:
            run_pairwise_xai(this_cfg, base_outdir)

    if pairwise_xai_flag:
        aggregate_all_metrics(base_outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    args = ap.parse_args()
    main(args.config, args.base_outdir)
