import numpy as np
import json

def compute_pairwise_metrics(out, cfg):

    print("🔢 Computing pairwise metrics...")
    pairwise_metrics = PairwiseMetrics(cfg)
    pairwise_results = pairwise_metrics.compute(out)



    pairwise_metrics_file = cfg.paths.metrics_dir / "pairwise_metrics.json"
    with open(pairwise_metrics_file, "w") as f:
        json.dump(pairwise_results, f, indent=2)
    print(f"✅ Pairwise metrics saved to {pairwise_metrics_file}")