import asyncio
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from .registry import discover_nla_config
from .client import NLAClient
from .interceptor import ThoughtInterceptor
from .tui import run_tui

async def chat(args):
    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device, trust_remote_code=True
    )
    mapping = discover_nla_config(model)
    if not mapping:
        print(f"Error: {args.model} architecture not supported.")
        return
    
    print(f"Found NLA: {mapping.nla_model_id} (Layer {mapping.extraction_layer})")
    client = NLAClient(mapping.nla_model_id, sglang_url=args.sglang_url)
    interceptor = ThoughtInterceptor(model, mapping.extraction_layer)
    
    prompt = input("\nEnter prompt: ")
    
    async def generator():
        ids = tok(prompt, return_tensors="pt").to(args.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=model.generate, kwargs=dict(**ids, streamer=streamer, max_new_tokens=150, do_sample=True, temperature=0.7))
        thread.start()
        for text in streamer:
            yield text
            await asyncio.sleep(0)

    with interceptor:
        await run_tui(generator, client, interceptor)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    c = sub.add_parser("chat")
    c.add_argument("--model", required=True)
    c.add_argument("--sglang-url", default="http://localhost:30000")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.cmd == "chat": asyncio.run(chat(args))

if __name__ == "__main__": main()
