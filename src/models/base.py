from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class BaseModel(ABC):
    """Abstract base interface for all models in the framework."""

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        pass
