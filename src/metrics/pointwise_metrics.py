import json
import pickle
import numpy as np
import torch as th
from pathlib import Path

from src.explainers.pointwise.utils import load_pointwise_explainer
from src.metrics.perturbation import (
    compute_insertion_curve,
    compute_deletion_curve,
    compute_AOPC,
)
from src.utils.loading import load_dataset, load_trained_model
from src.utils.config import make_outdir
from src.utils.plot_samples import plot_explainer_samples

def compute_pointwise_metrics(cfg, base_outdir):
    expl_cfg = cfg["pointwise_xai"]["explainer"]
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
    
    # ---------------------------------------------
    # 2. Perturbation curves
    # ---------------------------------------------
    # ins_curve = compute_insertion_curve(
    #     model, X_val, y_val, attributions, cfg
    # )
    # del_curve = compute_deletion_curve(
    #     model, X_val, y_val, attributions, cfg
    # )

    # np.save(expl_dir / f"{name}_insertion.npy", ins_curve)
    # np.save(expl_dir / f"{name}_deletion.npy", del_curve)

    # ---------------------------------------------
    # 3. AOPC
    # ---------------------------------------------
    # aopc = compute_AOPC(ins_curve, del_curve)

    # results[name] = {
    #     "AOPC": float(aopc),
    #     "insertion_final": float(ins_curve[-1]),
    #     "deletion_final": float(del_curve[-1]),
    # }

    # -------------------------------------------------
    # Save metrics
    # -------------------------------------------------
    # out_file = out_pointxai / "pointwise_metrics.json"
    # with open(out_file, "w") as f:
    #     json.dump(results, f, indent=2)

    # print(f"✅ Pointwise metrics saved to {out_file}")


