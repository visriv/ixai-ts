import torch
from captum.attr import GradientShap
from .base_explainer import BaseExplainer

class GradSHAPExplainer(BaseExplainer):
    """
    GradSHAP = Expected Integrated Gradients over random baselines
    """
    name = "GradSHAP"

    def __init__(
        self,
        model,
        n_samples: int = 20,
        stdev: float = 0.0,
    ):
        """
        Args:
            n_samples: number of random baselines
            stdev: std of Gaussian noise added to inputs (SmoothGrad-style)
        """
        super().__init__(model)
        self.n_samples = n_samples
        self.stdev = stdev

    def attribute(self, X):
        net = self.model
        device = next(net.parameters()).device
        net.eval()

        gs = GradientShap(net)

        if isinstance(X, torch.Tensor):
            X_t = X.detach().clone().to(device).float()
        else:
            X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # --------------------------------------------------
        # Baseline distribution
        # Here: zero baseline replicated n_samples times
        # You can replace this with dataset mean / samples
        # --------------------------------------------------
        baselines = torch.zeros(
            (self.n_samples,) + X_t.shape[1:],
            device=device,
            dtype=X_t.dtype,
        )

        # Compute target class ONCE (important)
        logits = net(X_t)
        target = logits.argmax(dim=1)

        # Captum expects baselines broadcastable to inputs
        attributions = gs.attribute(
            X_t,
            baselines=baselines,
            target=target,
            n_samples=self.n_samples,
            stdevs=self.stdev,
        )

        return attributions.detach().cpu().numpy()

Explainer = GradSHAPExplainer