import torch
import torch.nn as nn
from queue import Queue

class ThoughtInterceptor:
    """
    Middleware to intercept residual stream activations at a specific layer.
    Uses PyTorch forward hooks.
    """
    def __init__(self, model: nn.Module, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self.queue: Queue = Queue()
        self._handle = None

    def __enter__(self):
        base = getattr(self.model, "model", self.model)
        target = base.layers[self.layer_index] if hasattr(base, "layers") else base.h[self.layer_index]
        self._handle = target.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._handle: 
            self._handle.remove()
            self._handle = None

    def _hook(self, module: nn.Module, inputs: tuple, output: tuple | torch.Tensor):
        # The hidden states are the first element of the output tuple if it is a tuple
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Get the activation of the last token: shape -> [hidden_size]
        # We assume batch size 1 for TUI generation.
        if hidden_states.ndim == 3:
            last_token_activation = hidden_states[0, -1, :].detach().cpu()
        else:
            last_token_activation = hidden_states[-1, :].detach().cpu()
            
        self.queue.put(last_token_activation)