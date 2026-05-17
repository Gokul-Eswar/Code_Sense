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

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`interceptor.py`) provides the `ThoughtInterceptor` class, which is a specialized tool for 
# extracting internal neural activations (the "thoughts") from a Large Language Model during its forward pass.
#
# --- Imports ---
# - `torch`: PyTorch library, used for tensor manipulation.
# - `torch.nn`: Used for type hinting the `nn.Module` (the model).
# - `queue.Queue`: A thread-safe FIFO queue used to store captured activations so they can be read by another part of the app.
#
# --- Class: ThoughtInterceptor ---
#
# 1. `__init__(self, model, layer_index)`
#    - **Purpose**: Prepares the interceptor for a specific model and layer.
#    - **Parameters**:
#        - `model` (nn.Module): The LLM to intercept (e.g., a LlamaForCausalLM).
#        - `layer_index` (int): The specific residual stream layer to tap into (e.g., layer 20).
#    - **Variables**:
#        - `self.queue`: (type: `Queue`) The storage for captured activation vectors.
#        - `self._handle`: (type: `RemovableHandle` or `None`) A reference to the registered PyTorch hook, used to remove it later.
#
# 2. `__enter__(self)`
#    - **Purpose**: Enables the "Context Manager" pattern (using `with ...`). It automatically registers the hook.
#    - **Logic**:
#        - It first tries to find the internal "model" attribute (common in Hugging Face models).
#        - It then locates the target layer. It handles two common naming conventions: `.layers` (Llama/Gemma) and `.h` (GPT-2/Qwen).
#        - It calls `register_forward_hook(self._hook)` on the target layer.
#        - Returns `self` so it can be used in a `with` block.
#
# 3. `__exit__(self, *args)`
#    - **Purpose**: Ensures the hook is cleanly removed when the `with` block ends, preventing memory leaks or unwanted overhead.
#    - **Logic**: Calls `self._handle.remove()` if it exists.
#
# 4. `_hook(self, module, inputs, output)`
#    - **Purpose**: The actual function executed by PyTorch every time the target layer finishes its computation.
#    - **Parameters**:
#        - `module`: The layer module itself.
#        - `inputs`: The inputs to the layer (unused).
#        - `output`: The output of the layer (the hidden states).
#    - **Logic Flow**:
#        - **Step 1**: In many models, layers return a tuple (hidden_states, cache, etc.). We extract the first element (`output[0]`).
#        - **Step 2**: The hidden states usually have the shape `[batch_size, sequence_length, hidden_size]`.
#        - **Step 3**: Since we are streaming tokens one by one (batch size 1), we want the activation for the *most recent* token.
#        - **Step 4**: We use `[0, -1, :]` to get the last token's vector.
#        - **Step 5**: We `.detach()` it from the computation graph (to save memory) and move it to the `.cpu()`.
#        - **Step 6**: We `.put()` this vector into `self.queue` for the TUI or provider to consume.
#
