from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    name = "BaseExplainer"
    def __init__(self, model, **kwargs):
        self.model = model
    @abstractmethod
    def attribute(self, X, **kwargs):
        """Return attributions shaped like X [N, T, D] or [N, D, T]."""
        ...