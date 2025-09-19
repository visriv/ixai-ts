import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch as th
import sys
from itertools import product

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, make_outdir
from src.models.tcn import TCNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.utils import make_loader, train_classifier


# -------------------------------
# Dataset loader
# -------------------------------
def load_dataset(ds_cfg):
    """Load synthetic dataset based on config."""
    name = ds_cfg["name"].lower()
    if name == "var":
        from src.datasets.var import generate_var
        X, y, A = generate_var(**{k: v for k, v in ds_cfg.items() if k != "name"})
    elif name == "arfima":
        from src.datasets.arfima import generate_arfima
        X, y = generate_arfima(**{k: v for k, v in ds_cfg.items() if k != "name"})
        A = None
    elif name == "lorenz":
        from src.datasets.lorenz import generate_lorenz
        X, y = generate_lorenz(**{k: v for k, v in ds_cfg.items() if k != "name"})
        A = None
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, A, name


# -------------------------------
# Sweep handling
# -------------------------------
def expand_cfg(cfg):
    """
    Expand config into a list of configs if any param is a list.
    Backward compatible: if no lists, returns [cfg].
    """
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
# Core training
# -------------------------------
def run_training(cfg, base_outdir):
    out = make_outdir(base_outdir, cfg, nested=True)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    ds_cfg = dict(cfg['dataset'])
    X, y, A, ds_name = load_dataset(ds_cfg)

    # --- Train/Val split ---
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # --- Save datasets ---
    data_dir = Path("data") / ds_name
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.pkl", "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)
    with open(data_dir / "val.pkl", "wb") as f:
        pickle.dump({"X": X_val, "y": y_val}, f)

    # --- Stats ---
    def class_stats(y):
        uniq, counts = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(uniq, counts)}

    print(f"Dataset: {ds_name}")
    print(f"Train size: {len(y_train)} | Class balance: {class_stats(y_train)}")
    print(f"Val size:   {len(y_val)}   | Class balance: {class_stats(y_val)}")

    # --- Select model ---
    D = X.shape[2]
    C = len(np.unique(y))
    model_name = cfg['model']['name'].lower()
    if model_name == 'tcn':
        model = TCNClassifier(D, C, hidden=64, layers=2)
    elif model_name == 'lstm':
        model = LSTMClassifier(D, C, hidden=64)
    else:
        model = TransformerClassifier(D, C, d_model=64)

    # --- Train ---
    train_loader = make_loader(X_train, y_train,
                               batch=cfg['training']['batch_size'],
                               shuffle=True)
    val_loader = make_loader(X_val, y_val,
                             batch=cfg['training']['batch_size'],
                             shuffle=False)
    model, history = train_classifier(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=cfg['training']['epochs'],
        lr=1e-3,
        device='cuda',
        task_name=ds_name
    )

    # --- Save model + config ---
    th.save(model.state_dict(), out / "model.pt")
    with open(out / "meta.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Training complete. Saved to {out}")


# -------------------------------
# Main
# -------------------------------
def main(cfg_path, base_outdir):
    cfg = load_config(cfg_path)
    all_cfgs = expand_cfg(cfg)

    for this_cfg in all_cfgs:
        run_training(this_cfg, base_outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    args = ap.parse_args()
    main(args.config, args.base_outdir)
