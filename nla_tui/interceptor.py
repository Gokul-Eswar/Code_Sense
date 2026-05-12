import torch
import torch.nn as nn
from queue import Queue

class ThoughtInterceptor:
    def __init__(self, model: nn.Module, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self.queue = Queue()
        self._handle = None

    def __enter__(self):
        base = getattr(self.model, "model", self.model)
        target = base.layers[self.layer_index] if hasattr(base, "layers") else base.h[self.layer_index]
        self._handle = target.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._handle: self._handle.remove()

    def _hook(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        self.queue.put(h[:, -1, :].detach().cpu())
