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

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`remote.py`) implements the `RemoteHTTPProvider`, which allows the TUI to connect to a model 
# hosted on a remote server (via the NLA Sidecar). This is critical for users who don't have a powerful 
# GPU on their local machine.
#
# --- Imports ---
# - `asyncio`: Used for asynchronous network operations.
# - `httpx`: A modern HTTP client for Python, used here for its excellent support for async streaming.
# - `orjson`: Used for fast JSON parsing of the data received from the server.
# - `torch`: PyTorch, used to reconstruct the activation tensors from raw bytes.
# - `base64`: Used to decode the activation vectors sent over HTTP.
# - `numpy`: Used as an intermediate buffer for efficient memory conversion from bytes to torch tensors.
# - `.base.ActivationProvider`: The parent class defining the interface.
#
# --- Class: RemoteHTTPProvider ---
#
# 1. `__init__(self, remote_url)`
#    - **Purpose**: Initializes the provider with the server address.
#    - **Parameters**: `remote_url` (str): The base URL of the NLA Sidecar (e.g., `http://1.2.3.4:8080`).
#    - **Variables**:
#        - `self.remote_url`: The cleaned URL.
#        - `self._http_client`: (type: `httpx.AsyncClient`) A persistent HTTP client with a long (120s) timeout.
#
# 2. `get_model_info(self)`
#    - **Purpose**: Returns a identification string for the UI header.
#
# 3. `generate_stream(self, prompt, history, **kwargs)` (Async)
#    - **Purpose**: Sends the generation request to the remote server and streams the result back.
#    - **Logic Flow**:
#        - **Step 1**: Prepares the JSON `payload` containing the prompt and conversation history.
#        - **Step 2**: Opens a streaming POST request to the `/generate` endpoint using `self._http_client.stream`.
#        - **Step 3 (Streaming Loop)**: Iterates over the response lines using `aiter_lines()`.
#            - It looks for lines starting with `data: ` (Standard SSE format).
#            - If it finds `[DONE]`, it yields the final chunk and exits the loop.
#            - **Step 4 (Data Parsing)**:
#                - Parses the JSON data string into a dictionary.
#                - Extracts the `token`.
#            - **Step 5 (Activation Reconstruction)**:
#                - If `activation_b64` is present in the data:
#                    - Decodes the Base64 string into `raw_bytes`.
#                    - Uses `np.frombuffer` to interpret these bytes as an array of `float32`.
#                    - Converts the NumPy array into a `torch.Tensor` using `torch.from_numpy`.
#            - **Step 6**: Yields the dictionary containing the token and the reconstructed activation tensor.
#        - **Step 7 (Error Handling)**: If parsing or decoding fails, it yields an error token to inform the user.
#
