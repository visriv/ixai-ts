import json
import pickle
import numpy as np
import torch as th
from pathlib import Path

from src.explainers.pointwise.utils import load_pointwise_explainer
from src.metrics.perturbation import compute_perturbation_curves
from src.utils.loading import load_dataset, load_trained_model
from src.utils.config import make_outdir
from src.utils.plot_samples import plot_explainer_samples, plot_hughes_curves, plot_insertion_deletion_curves
from src.metrics.perturbation import compute_pointwise_metrics_from_curves
from src.metrics.pert_hugues import compute_hughes_curves, compute_hugues_metrics








def compute_pointwise_metrics(cfg, base_outdir):
    expl_cfg = cfg["pointwise"]["explainer"]
    name = expl_cfg["name"]
    params = expl_cfg.get("params", {})
    print(f"➡️  Explainer: {name}")
    out_train, out_pointxai, out_pairxai = make_outdir(base_outdir, cfg, nested=True)
    expl_dir = out_pointxai 
    expl_dir.mkdir(exist_ok=True)
    expl_path = expl_dir / f"{name}.pkl"

    device = "cuda" if th.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    (X_train, y_train), (X_val, y_val), A, data_dir = \
        load_dataset(ds_name=cfg["dataset"]["name"])

    X_val = th.from_numpy(X_val).float().to(device)
    y_val = th.from_numpy(y_val).long().to(device)


    if expl_path.exists():
        print(f"📂 Found cached attributions at {expl_path}, loading...")
        with open(expl_path, "rb") as f:
            attributions = pickle.load(f)
        plot_explainer_samples(
            X=X_val,
            y=y_val,
            attributions=attributions,
            expl_name=name,
            expl_dir=str(expl_dir),
            num_samples=cfg["pointwise"].get("num_plot_samples", 5),
        )
    else:
        print("📊 Computing pointwise metrics...")



        # -------------------------------------------------
        # Load model
        # -------------------------------------------------
        model = load_trained_model(cfg, base_outdir)
        model.eval().to(device)



        # curves_dir = metrics_dir / "pointwise_curves"
        # curves_dir.mkdir(exist_ok=True)

        results = {}

        # -------------------------------------------------
        # explainer
        # -------------------------------------------------


        ExplainerCls = load_pointwise_explainer(name)
        # explainer = ExplainerCls(**params)
        explainer = ExplainerCls(model=model, **params)


        # ---------------------------------------------
        # 1. Compute explanations
        # ---------------------------------------------
        attributions = explainer.attribute(
            X_val,
            # target=y if cfg["pointwise"]["output"]["target"] == "label" else None,
        )  # expected [N,T,D] or [N,D]

        with open(expl_path, "wb") as f:
            pickle.dump(attributions, f)

        plot_explainer_samples(
            X=X_val,
            y=y_val,
            attributions=attributions,
            expl_name=name,
            expl_dir=str(expl_dir),
            num_samples=cfg["pointwise"].get("num_plot_samples", 5),
        )
    


    curves_path = expl_dir / f"{name}_curves.npy"
    hugues_curves_path = expl_dir / f"{name}_hugues_curves.npy"
    if curves_path.exists() and hugues_curves_path.exists():
        print(f"📂 Found cached perturbation curves for {name}, loading...")
        curves = np.load(curves_path, allow_pickle=True).item()
        hugues_curves = np.load(hugues_curves_path, allow_pickle=True).item()
    else:
        print(f"📊 Computing perturbation curves for {name} ...")
        model = load_trained_model(cfg, base_outdir)
        model.eval().to(device)
        curves = compute_perturbation_curves(model, X_val, y_val, attributions, cfg)
        hugues_curves = compute_hughes_curves(model, X_val, y_val, attributions, cfg)
        np.save(expl_dir / f"{name}_curves.npy", curves)
        np.save(expl_dir / f"{name}_hugues_curves.npy", hugues_curves)


    plot_insertion_deletion_curves(
        curves,
        expl_name=name,
        expl_dir=str(expl_dir),
    )   
    metrics = compute_pointwise_metrics_from_curves(curves)
    with open(expl_dir / "pointwise_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


    # New set of metrics from Hugues et al.

    plot_hughes_curves(
        hugues_curves,
        expl_name=name,
        expl_dir=str(expl_dir),
    )   
    metrics = compute_hugues_metrics(hugues_curves)
    with open(expl_dir / "pointwise_metrics_hugues.json", "w") as f:
        json.dump(metrics, f, indent=2)