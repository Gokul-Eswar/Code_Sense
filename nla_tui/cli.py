import asyncio
import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from rich.console import Console
from rich.prompt import Prompt

from .registry import discover_nla_config
from .client import NLAClient
from .interceptor import ThoughtInterceptor
from .tui import run_tui, NLADashboard

console = Console()

async def chat(args):
    console.print(f"[bold green]Loading base model:[/bold green] {args.model} on {args.device}...")
    
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16 if "cuda" in args.device and not bnb_config else torch.float32,
            device_map=args.device, 
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    except Exception as e:
        console.print(f"[bold red]Failed to load base model {args.model}:[/bold red] {e}")
        return

    mapping = discover_nla_config(model)
    if not mapping:
        console.print(f"[bold red]Error:[/bold red] Architecture of {args.model} is not supported by current NLA mappings.")
        return
    
    console.print(f"[bold green]Discovered NLA Model:[/bold green] {mapping.nla_model_id} (Extracting at Layer {mapping.extraction_layer})")
    
    try:
        client = NLAClient(mapping.nla_model_id, sglang_url=args.sglang_url, device=args.device)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize NLA Client:[/bold red] {e}")
        return

    interceptor = ThoughtInterceptor(model, mapping.extraction_layer)
    
    history = []
    dashboard = NLADashboard()
    
    console.print("\n[bold cyan]Chat session started. Type 'exit' or 'quit' to end.[/bold cyan]\n")

    while True:
        try:
            prompt = Prompt.ask("[bold yellow]User[/bold yellow]")
            if prompt.strip().lower() in ("exit", "quit"):
                break
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt.strip():
            continue

        history.append({"role": "user", "content": prompt})
        dashboard.update(f"[bold yellow]User:[/bold yellow] {prompt}\n\n[bold green]Assistant:[/bold green] ")
        
        async def generator():
            formatted_prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            ids = tok(formatted_prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            
            gen_kwargs = dict(
                **ids, 
                streamer=streamer, 
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.7
            )
            
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            
            full_response = ""
            for text in streamer:
                full_response += text
                yield text
                await asyncio.sleep(0.01) # Yield control
            
            history.append({"role": "assistant", "content": full_response})
            dashboard.update("\n\n")

        with interceptor:
            await run_tui(generator, client, interceptor, dashboard=dashboard)

    console.print("\n[bold cyan]Session ended.[/bold cyan]")

def main():
    parser = argparse.ArgumentParser(description="NLA TUI - Visualize AI thoughts in real-time.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    c = sub.add_parser("chat", help="Start a chat session with visualization.")
    c.add_argument("--model", required=True, help="Base model ID (e.g., Qwen/Qwen2.5-7B-Instruct)")
    c.add_argument("--sglang-url", help="SGLang server URL (optional, defaults to local inference)")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (e.g., cuda, cpu)")
    c.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit (requires bitsandbytes)")
    c.add_argument("--load-in-8bit", action="store_true", help="Load base model in 8-bit (requires bitsandbytes)")
    
    args = parser.parse_args()
    if args.cmd == "chat":
        try:
            asyncio.run(chat(args))
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()
