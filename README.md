# ixai-ts

Interaction based eXplainable AI for Time Series (ixai-ts).

This repository contains code for:
- Synthetic dataset generation (VAR, ARFIMA, Lorenz, etc.)
- Training baseline models (TCN, LSTM, Transformer) on synthetic time series
- Computing interaction-based explanations using **Shapley–Taylor Interaction (STI)**
- Metrics and plots for locality, spectrum, and probabilistic baselines
- Experiment sweeps defined in config/var.yaml

## 📂 Project Structure
```
ixai-ts/
├── config/            # YAML configs for datasets and experiments
├── data/              # Generated synthetic datasets (train/val)
├── runs/              # Outputs: models, metrics, plots
├── scripts/           # Main entry scripts
│   ├── train_model.py
│   ├── compute_metrics.py
│   └── run_experiments.py
└── src/               # Source code (datasets, models, explainers, metrics, utils)
```

## 🚀 Usage

### 1. Train a Model
```bash
python scripts/train_model.py --config config/var.yaml --base_outdir runs
```

### 2. Compute Metrics & Plots
```bash
python scripts/compute_metrics.py --config config/var.yaml --ckpt runs/<exp_folder>/model.pt --base_outdir runs
```

### 3. Run All Experiments
```bash
python scripts/run_experiments.py --config config/var.yaml --base_outdir runs
```

This will train models, compute STI metrics, and aggregate results into `runs/all_metrics.csv`.

## ⚙️ Config (`config/var.yaml`)
- `dataset`: synthetic dataset parameters (num_series, seq_len, coeff_scale, noise)
- `model`: model architecture (`tcn`, `lstm`, `transformer`)
- `training`: batch size, epochs
- `experiment`: interaction parameters (`tau_max`, `num_permutations`)
- `sweeps`: defines multiple experiment variations (noise levels, permutations, coeff_scale, etc.)

## 📊 Outputs
Each experiment folder under `runs/` contains:
- `model.pt` → trained model checkpoint
- `history.json` → training history
- `metrics.json` → evaluation metrics
- `locality*.png`, `spectrum*.png` → plots

Aggregated results across sweeps are stored in:
```
runs/all_metrics.csv
```

## 🔑 Notes
- Use `tmux` or `nohup` for long experiments on servers.
- Datasets are saved in `data/<dataset>/train.pkl` and `val.pkl`.
- Metrics include accuracy, precision, recall, F1, AUROC, AUPRC, half-range, spectral centroid/flatness, etc.

---

```

