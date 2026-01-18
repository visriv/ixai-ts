import torch
from captum.attr import IntegratedGradients
from .base_explainer import BaseExplainer

class IGExplainer(BaseExplainer):
    name = "IG"

    def __init__(self, model,
                 steps, 
                 **kwargs):
        super().__init__(model)
        self.steps = steps
    def attribute(self, X):
        net = self.model
        device = next(net.parameters()).device
        net.eval()
        ig = IntegratedGradients(net)

        if isinstance(X, torch.Tensor):
            X_t = X.detach().clone().to(device).float()
        else:
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            
        baseline = torch.zeros_like(X_t)

        logits = net(X_t)
        target = logits.argmax(dim=1)

        net.train()
        # Integrated gradients attribution for that target
        attrs = ig.attribute(X_t, baselines=baseline, n_steps=self.steps, target=target)
        net.eval()
        return attrs.detach().cpu().numpy()

Explainer = IGExplainer
