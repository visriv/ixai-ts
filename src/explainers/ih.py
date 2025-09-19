import torch as th
from torch import nn

def integrated_hessians_pair(
    model: nn.Module, X: th.Tensor, u: tuple, v: tuple, steps: int = 20, device="cpu"
):
    """
    Approximate pairwise interaction by integrating second derivatives along straight line from baseline to X.
    u=(t,d), v=(t', d')
    """
    model.to(device)
    X = X.to(device).requires_grad_(True)
    T, D = X.shape
    base = th.zeros_like(X)

    alphas = th.linspace(0, 1, steps, device=device).view(-1,1,1)
    Xp = base.unsqueeze(0) + alphas * (X.unsqueeze(0) - base.unsqueeze(0))  # [S, T, D]

    logits = model(Xp)  # broadcasting; implement model to support [S, T, D] -> [S, C]
    if logits.dim()==2:
        logits = logits  # [S, C]
    p = th.softmax(logits, dim=-1)[..., 0]  # pick class 0 for demo; replace with target class

    # second derivative wrt x_u and x_v
    td1, dd1 = u
    td2, dd2 = v
    grads1 = th.autograd.grad(p.sum(), Xp, create_graph=True)[0]
    du = grads1[:, td1, dd1]
    d2 = th.autograd.grad(du.sum(), Xp, retain_graph=True)[0][:, td2, dd2]
    # integral approx (Riemann)
    ih_uv = d2.mean().item()
    return ih_uv
