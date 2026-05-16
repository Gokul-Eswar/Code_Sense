import asyncio
import httpx
import orjson
import torch
import base64
import numpy as np
from typing import AsyncGenerator, Dict, Any, Optional
from .base import ActivationProvider

class RemoteHTTPProvider(ActivationProvider):
    """
    Provider that connects to a remote NLA Sidecar or SGLang server.
    """
    def __init__(self, remote_url: str):
        self.remote_url = remote_url.rstrip("/")
        self._http_client = httpx.AsyncClient(timeout=120.0)

    def get_model_info(self) -> str:
        return f"Remote: {self.remote_url}"

    async def generate_stream(self, prompt: str, history: list, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        payload = {
            "prompt": prompt,
            "history": history,
            "max_new_tokens": kwargs.get("max_new_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7)
        }

        async with self._http_client.stream("POST", f"{self.remote_url}/generate", json=payload) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    yield {"token": "", "activation": None, "done": True}
                    break

                try:
                    data = orjson.loads(data_str)
                    token = data.get("token", "")

                    # Efficiently decode Base64 back into a tensor
                    activation = None
                    if "activation_b64" in data and data["activation_b64"]:
                        raw_bytes = base64.b64decode(data["activation_b64"])
                        # We assume float32 (the standard for these tensors in NLA)
                        arr = np.frombuffer(raw_bytes, dtype=np.float32)
                        activation = torch.from_numpy(arr.copy())

                    yield {
                        "token": token,
                        "activation": activation,
                        "done": False
                    }
                except Exception as e:
                    yield {"token": f"[Error: {e}]", "activation": None, "done": False}