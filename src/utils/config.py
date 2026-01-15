import yaml
from pathlib import Path
from copy import deepcopy
from itertools import product


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
        "dataset.autocorr_coeff": "gamma",
        "dataset.cross_coef": "alpha",
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
        "pairwise.interaction_method": "im",
        "pairwise.tau_max": "taumax",
        "pairwise.K": "K",
        "pairwise.num_permutations": "nperm"
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
    exp_tag   = build_tag("pairwise")

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


# utils for handing config sweeps and expansions


from copy import deepcopy
from itertools import product

def _set_by_path(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _get_by_path(d: dict, path: str):
    keys = path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _match_when(cfg: dict, when: dict) -> bool:
    # only equality checks (good enough for now)
    for kpath, expected in when.items():
        got = _get_by_path(cfg, kpath)
        if got != expected:
            return False
    return True

def expand_cfg(cfg: dict):
    """
    Works with your YAML:
      - models: [ {name: transformer, ...}, {name: lstm, ...} ]
      - sweeps: mix of old-style blocks and new-style {when, sweep} blocks
    Produces a global Cartesian product across all sweep params, with correct conditional logic.
    """
    base = deepcopy(cfg)

    # 1) models
    models = base.get("models", None)
    if not models or not isinstance(models, list):
        # fallback to single model for backward compatibility
        models = [base.get("model", {})]

    # 2) parse sweep blocks -> list of (when, sweep_dict)
    sweep_blocks = []
    sweeps = base.get("sweeps", {})
    if isinstance(sweeps, dict):
        for _, block in sweeps.items():
            if not isinstance(block, dict):
                continue

            # new style: has 'sweep' and optional 'when'
            if "sweep" in block or "when" in block:
                when = block.get("when", None)
                sweep_dict = block.get("sweep", {})
            else:
                # old style: direct mapping key -> list
                when = None
                sweep_dict = block

            if not isinstance(sweep_dict, dict):
                continue

            # keep even empty sweep_dict? no: ignore empties
            if len(sweep_dict) == 0:
                continue

            sweep_blocks.append((when, sweep_dict))

    # Helper: turn one sweep_dict into list of (kpath, values_list)
    def normalize_sweep_dict(sweep_dict):
        out = []
        for kpath, vals in sweep_dict.items():
            if not isinstance(vals, list):
                vals = [vals]
            out.append((kpath, vals))
        return out

    expanded = []

    # 3) expand per model
    for m in models:
        # start from base and attach model into cfg['model']
        cfg_m = deepcopy(base)
        cfg_m["model"] = deepcopy(m)
        if "models" in cfg_m:
            del cfg_m["models"]

        # 4) Build ONE global pool of parameters for this model:
        # For each sweep block:
        #   - if when matches (or when is None): include its parameters in the global product
        #   - else: block contributes "identity" (i.e., no params)
        global_keys = []
        global_vals = []

        for when, sweep_dict in sweep_blocks:
            if when is not None and not _match_when(cfg_m, when):
                # doesn't apply to this model → identity block
                continue

            kvs = normalize_sweep_dict(sweep_dict)
            for kpath, vals in kvs:
                # default section handling if user gives "num_permutations" etc.
                if "." not in kpath:
                    kpath = f"experiment.{kpath}"
                global_keys.append(kpath)
                global_vals.append(vals)

        # If no sweep params apply, still return cfg_m once
        if len(global_keys) == 0:
            cfg_m["_sweep_name"] = "nosweep"
            expanded.append(cfg_m)
            continue

        # 5) Global Cartesian product
        for combo in product(*global_vals):
            c = deepcopy(cfg_m)
            for kpath, val in zip(global_keys, combo):
                _set_by_path(c, kpath, val)
            c["_sweep_name"] = "cartesian"
            expanded.append(c)

    return expanded



def expand_cfg_old(cfg: dict):
    """
    Expand config into a list of configs:
      - expand over multiple models (cfg.models or cfg.model)
      - expand sweeps as SEPARATE blocks (VAR-2 independent from VAR-3 etc.)
      - optionally include a single 'baseline' run (either from base cfg, or VAR-1)
    """
    base = deepcopy(cfg)

    # Extract models (multi-model support)
    models = _get_models_list(base)

    # Helper: create configs for one sweep block (dict of keypath -> list)
    def expand_one_sweep_block(base_cfg: dict, sweep_pairs: dict):
        keys = []
        vals = []
        for keypath, v in sweep_pairs.items():
            if isinstance(v, list):
                keys.append(keypath)
                vals.append(v)
            else:
                # allow scalar overrides too
                keys.append(keypath)
                vals.append([v])

        out_cfgs = []
        for combo in product(*vals):
            c = deepcopy(base_cfg)
            for kpath, val in zip(keys, combo):
                # default section handling: if no dot, assume experiment.<k>
                if "." not in kpath:
                    kpath = f"experiment.{kpath}"
                _set_by_path(c, kpath, val)
            out_cfgs.append(c)
        return out_cfgs

    # 1) Build sweep blocks
    sweep_blocks = []
    sweeps = base.get("sweeps", None)
    if isinstance(sweeps, dict) and len(sweeps) > 0:
        # Treat each VAR-* as separate sweep
        for sweep_name, block in sweeps.items():
            if not isinstance(block, dict):
                continue

            when = block.get("when", None)
            sweep_pairs = block.get("sweep", {})

            if not isinstance(sweep_pairs, dict):
                continue

            sweep_blocks.append((sweep_name, when, sweep_pairs))


    # 2) Decide baseline behavior
    # If VAR-1 exists, treat that as baseline sweep block; else baseline is just base config.
    baseline_pairs = None
    for name, when, pairs in sweep_blocks:
        if name.lower() in ["var-1", "baseline", "base"]:
            baseline_pairs = pairs
            break

    # Remove baseline block from sweeps (so it doesn’t run twice)
    sweep_blocks_no_baseline = [(n, p) for (n, p) in sweep_blocks if p is not baseline_pairs]

    expanded = []

    # 3) Baseline run(s)
    # If baseline_pairs exist, expand it; otherwise run base cfg as baseline.
    baseline_cfgs = []
    if baseline_pairs is not None:
        baseline_cfgs = expand_one_sweep_block(base, baseline_pairs)
    else:
        baseline_cfgs = [deepcopy(base)]

    # 4) For each baseline cfg, expand models
    def attach_model(cfg0, model_dict):
        c = deepcopy(cfg0)
        c["model"] = deepcopy(model_dict)
        # keep models key out of final configs (avoid confusion)
        if "models" in c:
            del c["models"]
        return c

    # Baseline + models
    for bc in baseline_cfgs:
        for m in models:
            c = attach_model(bc, m)
            c["_sweep_name"] = "baseline"
            expanded.append(c)

    # 5) Run each sweep block independently (each also expanded over models)
    for sweep_name, when, pairs in sweep_blocks_no_baseline:
        sweep_cfgs = expand_one_sweep_block(base, pairs)

        for sc in sweep_cfgs:
            for m in models:
                c = attach_model(sc, m)

                # APPLY FILTER HERE
                if when is not None and not _match_when(c, when):
                    continue

                c["_sweep_name"] = sweep_name
                expanded.append(c)


    # If no sweeps at all, still expand over models
    if not sweep_blocks:
        expanded = []
        for m in models:
            c = attach_model(base, m)
            c["_sweep_name"] = "nosweep"
            expanded.append(c)

    return expanded
