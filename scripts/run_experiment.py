import argparse
import subprocess
import os
from pathlib import Path
import sys
import json
import pandas as pd

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config

def run_all(cfg_path, base_outdir):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)  # project root

    # --- Train all sweeps ---
    subprocess.check_call(
        ["python", "scripts/train_model.py", "--config", cfg_path, "--base_outdir", base_outdir],
        env=env
    )

    # --- Compute metrics for all sweeps ---
    subprocess.check_call(
        ["python", "scripts/compute_metrics.py", "--config", cfg_path, "--base_outdir", base_outdir],
        env=env
    )

    # --- Aggregate metrics into a single CSV ---
    outdir = Path(base_outdir)
    metrics_files = list(outdir.rglob("metrics*.json"))

    rows = []
    for mf in metrics_files:
        with open(mf, "r") as f:
            metrics = json.load(f)
        # Add experiment ID = relative path
        exp_id = str(mf.parent.relative_to(outdir))
        rows.append({"exp_id": exp_id, **metrics})

    if rows:
        df = pd.DataFrame(rows)
        csv_path = outdir / "all_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Aggregated metrics written to {csv_path}")
    else:
        print("⚠️ No metrics files found to aggregate.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_outdir", default="runs")
    args = ap.parse_args()
    run_all(args.config, args.base_outdir)
