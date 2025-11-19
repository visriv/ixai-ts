import torch as th
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
import numpy as np


def make_loader(X, y, batch=64, shuffle=True):
    X = th.tensor(X, dtype=th.float32)
    y = th.tensor(y, dtype=th.long)

    if X.dim() == 2:
        # [N, D] -> [N, 1, D] (add time dimension)
        X = X.unsqueeze(1)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def evaluate_classifier(model, loader, device="cpu"):
    """Run evaluation and compute metrics."""
    model.eval()
    all_y, all_p, all_s = [], [], []  # labels, preds, scores
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    with th.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)           # [B, T, C]
            B, T, C = logits.shape

            loss = loss_fn(
                logits.view(B * T, C),   # [B*T, C]
                yb.view(B * T),          # [B*T]
            )
            losses.append(loss.item())

            probs = th.softmax(logits, dim=-1)      # [B, T, C]
            preds = probs.argmax(dim=-1)            # [B, T]

            # --- FLATTEN for sklearn ---
            y_np = yb.cpu().numpy().reshape(-1)          # [B*T]
            p_np = preds.cpu().numpy().reshape(-1)       # [B*T]
            s_np = probs[..., 1].cpu().numpy().reshape(-1) if probs.shape[-1] > 1 \
                  else probs[..., 0].cpu().numpy().reshape(-1)

            all_y.append(y_np)
            all_p.append(p_np)
            all_s.append(s_np)

    # concat over batches -> 1D vectors
    y_true = np.concatenate(all_y, axis=0)   # [N_total]
    y_pred = np.concatenate(all_p, axis=0)
    y_score = np.concatenate(all_s, axis=0)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    else:
        auroc = float("nan")
        auprc = float("nan")

    val_loss = np.mean(losses)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "val_loss": val_loss,
    }


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    device="cpu",
    task_name="classification",
):
    model.to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []

    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(
            train_loader, desc=f"[{task_name}] train ep {ep+1}/{epochs}"
        ):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            logits = model(xb)               # [B, T, C]
            B, T, C = logits.shape
            loss = loss_fn(
                logits.view(B * T, C),       # [B*T, C]
                yb.view(B * T),              # [B*T]
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)
        log = {"epoch": ep + 1, "train_loss": round(train_loss, 4)}

        # --- Validation ---
        if val_loader is not None:
            metrics = evaluate_classifier(model, val_loader, device=device)
            # round all float metrics to 4 decimals
            metrics = {
                k: round(v, 4) if isinstance(v, (float, np.floating)) else v
                for k, v in metrics.items()
            }
            log.update(metrics)

        history.append(log)
        print(f"Epoch {ep+1}: {log}")

    return model, history
