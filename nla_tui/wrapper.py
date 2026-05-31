import asyncio
from typing import Optional, Any
from .providers.local import LocalHFProvider
from .client import NLAClient
from .tui import NLATextualApp
from .filter import ThoughtFilter

class NLAWrapper:
    """
    A high-level wrapper to easily add NLA thought visualization to any model generation.
    """
    def __init__(self, model_id: str, device: str = "cpu", sglang_url: Optional[str] = None, **kwargs):
        self.provider = LocalHFProvider(model_id, device=device, **kwargs)
        self.client = NLAClient(self.provider.mapping.nla_model_id, sglang_url=sglang_url, device=device)
        self.app = NLATextualApp(model_info=self.provider.get_model_info())

    async def run_chat_ui(self):
        history = []
        thought_filter = ThoughtFilter()

        async def proc_thought(act):
            try:
                thought = await self.client.get_thought(act)
                self.app.call_from_thread(self.app.write_thought, thought)
            except Exception as e:
                self.app.call_from_thread(self.app.write_thought, f"[red]Error:[/red] {e}")

        async def run_generation(prompt):
            history.append({"role": "user", "content": prompt})
            thought_filter.reset()
            stream = self.provider.generate_stream(prompt, history)
            
            full_response = ""
            async for chunk in stream:
                if chunk["done"]:
                    break
                
                token = chunk["token"]
                activation = chunk["activation"]
                
                if token:
                    full_response += token
                    self.app.call_from_thread(self.app.write_token, token)
                
                if activation is not None:
                    if thought_filter.should_translate(token, activation):
                        asyncio.create_task(proc_thought(activation))
                    
                await asyncio.sleep(0.001)
                
            history.append({"role": "assistant", "content": full_response})
            self.app.call_from_thread(self.app.write_token, "\n\n")

        self.app.gen_fn = run_generation
        await self.app.run_async()

def wrap_model(model_id: str, **kwargs) -> NLAWrapper:
    """Utility function to wrap a model by its ID."""
    return NLAWrapper(model_id, **kwargs)

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`wrapper.py`) provides a simplified, high-level interface for developers who want to add 
# NLA thought visualization to their models with minimal setup. It bundles the provider, client, and UI.
#
# --- Imports ---
# - `asyncio`: Used for non-blocking I/O and task management.
# - `typing.Optional, Any`: Used for type hinting.
# - `.providers.local.LocalHFProvider`: Used to run models locally.
# - `.client.NLAClient`: Used to verbalize activations.
# - `.tui.NLATextualApp`: The TUI display engine.
#
# --- Class: NLAWrapper ---
#
# 1. `__init__(self, model_id, device="cpu", sglang_url=None, **kwargs)`
#    - **Purpose**: Sets up all the necessary components for an NLA-enabled session.
#    - **Parameters**:
#        - `model_id` (str): The HF model ID.
#        - `device` (str): Target hardware.
#        - `sglang_url` (Optional[str]): Remote verbalizer URL.
#        - `**kwargs`: Passed directly to the `LocalHFProvider` (e.g., quantization settings).
#    - **Variables**:
#        - `self.provider`: (type: `LocalHFProvider`) Manages the local model.
#        - `self.client`: (type: `NLAClient`) Manages the verbalizer connection.
#        - `self.app`: (type: `NLATextualApp`) The UI instance.
#
# 2. `run_chat_ui(self)` (Async)
#    - **Purpose**: Starts the full interactive TUI chat session.
#    - **Variables**:
#        - `history`: (type: `list`) Stores the conversation messages.
#    - **Logic Flow**:
#        - **Internal Function: `proc_thought(act)`**:
#            - Takes an activation vector, asks the client for a "thought", and writes it to the UI's mind pane.
#        - **Internal Function: `run_generation(prompt)`**:
#            - Adds user prompt to history.
#            - Starts the model generation stream.
#            - Iterates over chunks:
#                - If a token is received, writes it to the output pane.
#                - If an activation is received, spawns a `proc_thought` task.
#            - Finally, adds the full assistant response to history.
#        - Sets `self.app.gen_fn = run_generation` and executes `self.app.run_async()`.
#
# --- Functions ---
#
# 1. `wrap_model(model_id, **kwargs)`
#    - **Purpose**: A convenience factory function.
#    - **Return**: A new instance of `NLAWrapper`.
#
