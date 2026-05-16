import asyncio
from typing import Optional, Any
from .providers.local import LocalHFProvider
from .client import NLAClient
from .tui import NLATextualApp

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

        async def proc_thought(act):
            try:
                thought = await self.client.get_thought(act)
                self.app.call_from_thread(self.app.write_thought, thought)
            except Exception as e:
                self.app.call_from_thread(self.app.write_thought, f"[red]Error:[/red] {e}")

        async def run_generation(prompt):
            history.append({"role": "user", "content": prompt})
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
                    asyncio.create_task(proc_thought(activation))
                    
                await asyncio.sleep(0.001)
                
            history.append({"role": "assistant", "content": full_response})
            self.app.call_from_thread(self.app.write_token, "\n\n")

        self.app.gen_fn = run_generation
        await self.app.run_async()

def wrap_model(model_id: str, **kwargs) -> NLAWrapper:
    """Utility function to wrap a model by its ID."""
    return NLAWrapper(model_id, **kwargs)