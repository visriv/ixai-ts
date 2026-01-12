# src/explainers/pointwise/utils.py

import importlib

def load_pointwise_explainer(name: str):
    """
    Loads explainer class from:
    src.explainers.pointwise.<name>.py
    """
    module_path = f"src.explainers.pointwise.{name}"
    module = importlib.import_module(module_path)

    if not hasattr(module, "Explainer"):
        raise ValueError(f"{module_path} must expose class Explainer")

    return module.Explainer
