#!/usr/bin/env python3
"""
Visualize datasets saved as pickles (train.pkl / val.pkl).
Each pickle contains a tuple (X, y) with shapes:
  X: [N, T, D], y: [N]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Ensure dict format with keys "X" and "y"
    if isinstance(data, dict) and "X" in data and "y" in data:
        return data["X"], data["y"]
    elif isinstance(data, tuple) or isinstance(data, list):
        return data  # (X, y)
    else:
        raise ValueError(f"Unexpected format in {path}")



def visualize_dataset(X, y, outdir=None, num_samples=5, split_name="train"):
    N, T, D = X.shape
    print(f"[{split_name}] Dataset shape: {X.shape}, Labels shape: {y.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"[{split_name}] Classes:", dict(zip(classes, counts)))

    outdir = Path(outdir) / split_name if outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # ---- Plot few random samples ----
    idxs = np.random.choice(N, size=min(num_samples, N), replace=False)
    for idx in idxs:
        plt.figure(figsize=(10, 4))
        for d in range(D):
            plt.plot(X[idx, :, d], label=f"Feature {d}")
        plt.title(f"{split_name} sample {idx} | Label={y[idx]}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        if outdir:
            plt.savefig(outdir / f"sample_{idx}.png")
            plt.close()
        else:
            plt.show()

    # ---- Distribution of labels ----
    plt.figure()
    sns.countplot(x=y)
    plt.title(f"{split_name} Class Distribution")
    if outdir:
        plt.savefig(outdir / "label_distribution.png")
        plt.close()
    else:
        plt.show()

    # ---- Feature statistics ----
    mean_vals = X.mean(axis=(0, 1))  # [D]
    std_vals  = X.std(axis=(0, 1))   # [D]
    plt.figure(figsize=(8, 4))
    plt.errorbar(range(D), mean_vals, yerr=std_vals, fmt='o')
    plt.title(f"{split_name} Feature Means ± Std")
    plt.xlabel("Feature index")
    plt.ylabel("Value")
    if outdir:
        plt.savefig(outdir / "feature_stats.png")
        plt.close()
    else:
        plt.show()

    # ---- Feature correlation heatmap ----
    X_flat = X.reshape(-1, D)
    corr = np.corrcoef(X_flat, rowvar=False)  # [D,D]
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title(f"{split_name} Feature Correlation Heatmap")
    if outdir:
        plt.savefig(outdir / "feature_correlation.png")
        plt.close()
    else:
        plt.show()

    print(f"✅ {split_name} visualization complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train.pkl")
    parser.add_argument("--val", type=str, required=False, help="Path to val.pkl")
    parser.add_argument("--outdir", type=str, default="runs/eda", help="Directory to save plots")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to plot")
    args = parser.parse_args()

    # Train split
    X_train, y_train = load_pickle(args.train)
    print(X_train)
    print(y_train)
    visualize_dataset(X_train, y_train, outdir=args.outdir, num_samples=args.num_samples, split_name="train")

    # Val split (optional)
    if args.val:
        X_val, y_val = load_pickle(args.val)
        visualize_dataset(X_val, y_val, outdir=args.outdir, num_samples=args.num_samples, split_name="val")


if __name__ == "__main__":
    main()
