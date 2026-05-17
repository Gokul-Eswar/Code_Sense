import asyncio
import argparse
import uvicorn
import orjson
import base64
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from .providers.local import LocalHFProvider

app = FastAPI(title="NLA Activation Sidecar")
provider = None

@app.post("/generate")
async def generate(request: Request):
    global provider
    body = await request.json()
    prompt = body.get("prompt", "")
    history = body.get("history", [])
    max_new_tokens = body.get("max_new_tokens", 256)
    temperature = body.get("temperature", 0.7)
    
    async def event_generator():
        async for chunk in provider.generate_stream(prompt, history, max_new_tokens=max_new_tokens, temperature=temperature):
            if chunk["done"]:
                yield "data: [DONE]\n\n"
                break
            
            # Use Base64 for activation tensors to reduce payload size
            activation_b64 = None
            activation = chunk["activation"]
            if activation is not None:
                # Convert to numpy and then to bytes for efficient encoding
                activation_b64 = base64.b64encode(activation.numpy().tobytes()).decode("ascii")
            
            data = {
                "token": chunk["token"],
                "activation_b64": activation_b64
            }
            yield f"data: {orjson.dumps(data).decode()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def main():
    parser = argparse.ArgumentParser(description="NLA Sidecar - Expose hidden states over HTTP.")
    parser.add_argument("--model", required=True, help="HF model ID to load.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on.")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    
    global provider
    print(f"Loading model {args.model} for Sidecar...")
    provider = LocalHFProvider(args.model, device=args.device, load_in_4bit=args.load_in_4bit)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`sidecar.py`) implements a FastAPI-based server that "exposes" a model's internal states over HTTP.
# This allows a user to run a heavy LLM on a GPU-enabled server while viewing the TUI on a different machine.
#
# --- Imports ---
# - `asyncio`: Used for managing asynchronous event loops.
# - `argparse`: Used for command-line argument parsing.
# - `uvicorn`: The ASGI server used to host the FastAPI application.
# - `orjson`: A high-performance JSON library.
# - `base64`: Used to encode raw binary tensor data into ASCII strings for HTTP transport.
# - `fastapi`: The web framework used to build the API.
# - `.providers.local.LocalHFProvider`: The local model runner used as the backend for the sidecar.
#
# --- Global Variables ---
# - `app`: (type: `FastAPI`) The main application instance.
# - `provider`: (type: `LocalHFProvider` or `None`) Global reference to the model provider, initialized in `main()`.
#
# --- API Endpoints ---
#
# 1. `@app.post("/generate")`
#    - **Purpose**: Accepts a prompt and streams back tokens paired with their internal activation vectors.
#    - **Parameters**: `request` (type: `Request`): The FastAPI request object containing JSON body.
#    - **Logic Flow**:
#        - **Step 1**: Parses `prompt`, `history`, `max_new_tokens`, and `temperature` from the request body.
#        - **Step 2**: Defines an internal `async def event_generator()` to handle the Server-Sent Events (SSE) stream.
#        - **Step 3 (Inside Generator)**: Calls `provider.generate_stream(...)` to get an async iterator.
#        - **Step 4 (Inside Loop)**: For each chunk generated:
#            - If `activation` exists, it converts the PyTorch tensor to a NumPy array, then to raw bytes, and finally to a Base64 string.
#            - Bundles the `token` and `activation_b64` into a dictionary.
#            - Yields the data formatted as an SSE message (`data: ...\n\n`).
#        - **Step 5**: Returns a `StreamingResponse` using the generator.
#
# --- Functions ---
#
# 1. `main()`
#    - **Purpose**: The entry point when running the sidecar as a standalone script.
#    - **Logic Flow**:
#        - Defines arguments for `--model`, `--device`, `--port`, and quantization.
#        - Initializes the global `provider` using `LocalHFProvider`.
#        - Starts the `uvicorn` server on `0.0.0.0` at the specified port.
#
# --- Main Execution Block ---
# - Calls `main()` if the script is run directly.
#
