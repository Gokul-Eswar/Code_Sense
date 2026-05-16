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