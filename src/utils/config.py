import yaml
from pathlib import Path

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

from pathlib import Path

def make_outdir(base_outdir, cfg, nested=True):
    """
    Create output directory encoding ALL relevant config params,
    with internal whitelist / blacklist and key shortening.
    """
    base_outdir = Path(base_outdir)

    # -----------------------------
    # Static rules (EDIT HERE ONLY)
    # -----------------------------

    # Keys to always ignore (noise, bookkeeping, paths, seeds, etc.)
    BLACKLIST = {
        "seed", "device", "log_dir", "outdir", "num_workers",
        "save", "debug", "verbose", "wandb", "mlflow", "all_times",
        "label_mode", "periodic_1", "periodic_2", 
        "noise", "name"
    }


    # Short aliases for readability
    KEY_ALIASES = {
        # dataset
        "dataset.num_samples": "n",
        "dataset.num_series": "d",
        "dataset.num_features": "d",
        "dataset.seq_len": "L",
        "dataset.window_size": "w",
        "dataset.num_interactions": "m",
        "dataset.autocorr_coeff": "rho",
        "dataset.cross_coef": "gamma",
        # model
        "model.name": "m",
        "model.d_model": "dm",
        "model.hidden": "h",
        "model.layers": "ly",
        "model.n_heads": "H",

        # training
        "training.epochs": "ep",
        "training.batch_size": "bs",
        "training.lr": "lr",
        "training.weight_decay": "wd",

        # experiment
        "experiment.interaction_method": "im",
        "experiment.tau_max": "taumax",
        "experiment.K": "K",
        "experiment.num_permutations": "nperm"
    }

    # -----------------------------
    # Helpers
    # -----------------------------

    def flatten(d, parent=""):
        """Recursively flatten dict: a.b.c -> value"""
        out = {}
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    def normalize(v):
        """Compact string for values"""
        if isinstance(v, float):
            return f"{v:.2g}"
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(map(str, v)) + "]"
        return str(v)

    def keep(k):
        if k.split(".")[-1] in BLACKLIST:
            return False
        return True

    # -----------------------------
    # Build tags per section
    # -----------------------------

    flat = flatten(cfg)

    def build_tag(prefix):
        items = []
        for k in sorted(flat.keys()):
            if not k.startswith(prefix + "."):
                continue
            if not keep(k):
                continue
            short = KEY_ALIASES.get(k, k.replace(prefix + ".", ""))
            items.append(f"{short}{normalize(flat[k])}")
        return "_".join(items)

    ds_name  = cfg["dataset"]["name"]
    ds_tag   = build_tag("dataset")
    model_name = cfg["model"]["name"]
    model_tag = build_tag("model")
    train_tag = build_tag("training")
    exp_tag   = build_tag("experiment")

    # -----------------------------
    # Final path
    # -----------------------------

    if nested:
        out = (
            base_outdir /
            f"{ds_name}_{ds_tag}" /
            f"{model_name}_{model_tag}" /
            train_tag /
            exp_tag
        )
    else:
        out = base_outdir / "_".join(
            x for x in [ds_name, ds_tag, model_tag, train_tag, exp_tag] if x
        )

    out.mkdir(parents=True, exist_ok=True)
    return out
