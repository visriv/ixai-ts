import torch
from captum.attr import ShapleyValueSampling
from .base_explainer import BaseExplainer


class ShapleySamplingExplainer(BaseExplainer):
    """
    First-order Shapley Value approximation via permutation sampling.
    """
    name = "ShapleySampling"

    def __init__(
        self,
        model,
        n_samples: int = 100,
        baseline: str = "zero",
    ):
        """
        Args:
            n_samples: number of permutation samples
            baseline: zero | mean
        """
        super().__init__(model)
        self.n_samples = n_samples
        self.baseline = baseline

    def attribute(self, X):
        net = self.model
        device = next(net.parameters()).device
        net.eval()

        sv = ShapleyValueSampling(net)

        if isinstance(X, torch.Tensor):
            X_t = X.detach().clone().to(device).float()
        else:
            X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # -----------------------------
        # Baseline construction
        # -----------------------------
        if self.baseline == "mean":
            baselines = X_t.mean(dim=0, keepdim=True)
        else:
            baselines = torch.zeros_like(X_t)

        # -----------------------------
        # Target class (fixed!)
        # -----------------------------
        logits = net(X_t)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        target = logits.argmax(dim=1)

        # -----------------------------
        # Attribution
        # -----------------------------
        attrs = sv.attribute(
            X_t,
            baselines=baselines,
            target=target,
            n_samples=self.n_samples,
        )

        return attrs.detach().cpu().numpy()


Explainer = ShapleySamplingExplainer

