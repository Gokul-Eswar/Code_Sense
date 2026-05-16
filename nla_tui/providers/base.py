from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
import torch

class ActivationProvider(ABC):
    """
    Abstract base class for providing activations and tokens from a base model.
    """
    @abstractmethod
    async def generate_stream(self, prompt: str, history: list, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yields dictionaries containing:
        - "token": str (the generated token)
        - "activation": Optional[torch.Tensor] (the residual stream vector)
        - "done": bool (whether generation is finished)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        """Returns a string describing the model and its location."""
        pass