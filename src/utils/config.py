import yaml
from pathlib import Path

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_outdir(base_outdir, cfg, nested=True):
    """
    Create output directory that encodes dataset, model, training & experiment hyperparams.
    """
    base_outdir = Path(base_outdir)

    # Dataset
    ds_name = cfg['dataset']['name']
    ds_tag  = f"n{cfg['dataset'].get('num_samples','NA')}_s{cfg['dataset'].get('num_series','NA')}_L{cfg['dataset'].get('seq_len','NA')}"

    # Model
    model_name = cfg['model']['name'] + '_d_model' + str(cfg["model"].get("d_model", "")) + '_layers' + str(cfg["model"].get("layers", ""))
    model_tag  = "_".join([f"{k}{v}" for k,v in cfg['model'].items() if k != "name"])

    # Training
    train_tag = f"ep{cfg['training'].get('epochs','NA')}_bs{cfg['training'].get('batch_size','NA')}"

    # Experiment
    exp_tag = "_".join([f"{k}{v}" for k,v in cfg['experiment'].items()])

    if nested:
        out = base_outdir / ds_name / model_name / train_tag / exp_tag
    else:
        out = base_outdir / f"{ds_name}_{model_name}_{ds_tag}_{model_tag}_{train_tag}_{exp_tag}"

    out.mkdir(parents=True, exist_ok=True)
    return out
