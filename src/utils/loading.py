from pathlib import Path
import pickle
import torch as th
import numpy as np
from src.utils.config import make_outdir
from src.models.tcn import TCNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


def select_model(cfg_model, D, C, all_times=False):
    name = str(cfg_model["name"]).lower()
    if name == "tcn":
        return TCNClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)), layers=int(cfg_model.get("layers", 2)))
    elif name == "lstm":
        return LSTMClassifier(D, C, hidden=int(cfg_model.get("hidden", 64)))
    else:
        return TransformerClassifier(D, C, d_model=int(cfg_model.get("d_model", 64)), all_times=all_times)



# ---------------------------------
# Dataset utils
# ---------------------------------
def load_dataset(ds_cfg):
    name = ds_cfg["name"].lower()
    params = {k: v for k, v in ds_cfg.items() if k != "name"}
    if name == "var":
        from src.datasets.var import generate_var
        X, y, A = generate_var(**params)

    elif name == "var_local" or name == "var_local_debug":
        from src.datasets.var_planted import generate_var
        X, y, A = generate_var(**params)
    elif name == "cltts" or name == "cltts_debug":
        from src.datasets.cltts import generate_cltts
        X, y, A = generate_cltts(**params)
    elif name == "arfima":
        from src.datasets.arfima import generate_arfima
        X, y = generate_arfima(**params)
        A = None
    elif name == "lorenz" or name == "lorenz_debug" or name == "lorenz_long":
        from src.datasets.lorenz import generate_lorenz
        X, y, A = generate_lorenz(**params)
    elif name == "ih_multi" or name == "ih_multi_debug":
        from src.datasets.ih_multi import generate_ih_multi
        X, y, A = generate_ih_multi(**params)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, A, name


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





def load_trained_model(cfg, base_outdir):
    """
    Load a trained model corresponding to cfg and base_outdir.

    Assumes:
      - model checkpoint is saved as: <outdir>/model.pt
      - dataset pickles exist OR dataset can be reloaded
    """

    # --------------------------------------------------
    # Resolve output directory (same logic as training)
    # --------------------------------------------------
    out = make_outdir(
        base_outdir=base_outdir,
        cfg=cfg,
        nested=True,   # must match training
    )

    ckpt_path = out / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    # --------------------------------------------------
    # Load dataset metadata (to infer D, C)
    # --------------------------------------------------
    ds_name = cfg["dataset"]["name"]

    (train, val, _) = load_pickles_if_exist(ds_name)

    if train is not None:
        X_train, y_train = train
    else:
        # Fallback: regenerate dataset (rare, but safe)
        X, y, *_ = load_dataset(cfg["dataset"])
        X_train, y_train = X, y

    D = X_train.shape[-1]
    C = int(len(np.unique(y_train)))

    # --------------------------------------------------
    # Reconstruct model architecture
    # --------------------------------------------------
    model = select_model(
        cfg_model=cfg["model"],
        D=D,
        C=C,
        all_times=cfg["dataset"].get("all_times", False),
    )

    # --------------------------------------------------
    # Load weights (always map to CPU first)
    # --------------------------------------------------
    state = th.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    return model