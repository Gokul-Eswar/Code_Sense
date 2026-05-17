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

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`base.py`) defines the interface (contract) that all activation providers must follow.
# It ensures that whether the model is running locally, remotely, or in some other configuration,
# the TUI and Wrapper can interact with it in a consistent way.
#
# --- Imports ---
# - `abc.ABC`, `abc.abstractmethod`: Used to define Abstract Base Classes in Python.
# - `typing.AsyncGenerator`, `Dict`, `Any`, `Optional`: Used for type hinting.
# - `torch`: PyTorch, used for referencing the `torch.Tensor` type for activations.
#
# --- Class: ActivationProvider (Abstract Base Class) ---
#
# 1. `generate_stream(self, prompt, history, **kwargs)` (Abstract Async Method)
#    - **Purpose**: This is the core method that must be implemented by subclasses. It should handle the actual 
#      model inference and stream the results back.
#    - **Parameters**:
#        - `prompt` (str): The current user input.
#        - `history` (list): The list of previous chat messages.
#        - `**kwargs`: Additional generation parameters (like `temperature`).
#    - **Yield Value**: An `AsyncGenerator` yielding dictionaries.
#        - `token`: (type: `str`) The actual text token generated.
#        - `activation`: (type: `Optional[torch.Tensor]`) The neural vector from the residual stream.
#        - `done`: (type: `bool`) A flag indicating if this is the final chunk.
#
# 2. `get_model_info(self)` (Abstract Method)
#    - **Purpose**: Returns a descriptive string about the backend (e.g., "Local: llama-3" or "Remote: sidecar-8080").
#    - **Return**: A string.
#
