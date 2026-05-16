import asyncio
import argparse
import sys
import torch
from rich.console import Console

from .providers.local import LocalHFProvider
from .providers.remote import RemoteHTTPProvider
from .client import NLAClient
from .tui import NLATextualApp
from .filter import ThoughtFilter

console = Console()

async def chat(args):
    if args.remote_url:
        console.print(f"[bold green]Connecting to remote provider:[/bold green] {args.remote_url}...")
        provider = RemoteHTTPProvider(args.remote_url)
        nla_model_id = args.nla_model
        if not nla_model_id:
            console.print("[bold red]Error:[/bold red] --nla-model is required when using --remote-url.")
            return
    else:
        console.print(f"[bold green]Loading local provider:[/bold green] {args.model}...")
        provider = LocalHFProvider(
            args.model, 
            device=args.device, 
            load_in_4bit=args.load_in_4bit, 
            load_in_8bit=args.load_in_8bit
        )
        nla_model_id = provider.mapping.nla_model_id

    console.print(f"[bold green]NLA Verbalizer:[/bold green] {nla_model_id}")
    
    try:
        client = NLAClient(nla_model_id, sglang_url=args.sglang_url, device=args.device)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize NLA Client:[/bold red] {e}")
        return

    app = NLATextualApp(model_info=provider.get_model_info())
    history = []
    thought_filter = ThoughtFilter()
    sem = asyncio.Semaphore(5) # Max 5 concurrent verbalizer requests

    async def process_generation(prompt):
        history.append({"role": "user", "content": prompt})
        thought_filter.reset()
        
        # Generator from provider
        stream = provider.generate_stream(prompt, history)
        
        full_response = ""
        async for chunk in stream:
            if chunk["done"]:
                break
            
            token = chunk["token"]
            activation = chunk["activation"]
            
            if token:
                full_response += token
                app.call_from_thread(app.write_token, token)
            
            if activation is not None:
                # Hybrid Smart Filtering
                if thought_filter.should_translate(token, activation):
                    asyncio.create_task(proc_thought(activation))
                
            await asyncio.sleep(0.001)
            
        history.append({"role": "assistant", "content": full_response})
        app.call_from_thread(app.write_token, "\n\n")

    async def proc_thought(act):
        async with sem:
            try:
                thought = await client.get_thought(act)
                app.call_from_thread(app.write_thought, thought)
            except Exception as e:
                app.call_from_thread(app.write_thought, f"[red]Error:[/red] {e}")

    app.gen_fn = process_generation
    await app.run_async()

def main():
    parser = argparse.ArgumentParser(description="NLA TUI - Visualize AI thoughts in real-time.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    c = sub.add_parser("chat", help="Start a chat session with visualization.")
    c.add_argument("--model", help="Base model ID (required for local mode)")
    c.add_argument("--remote-url", help="URL of a remote NLA Sidecar or SGLang server.")
    c.add_argument("--nla-model", help="NLA Verbalizer model ID (required for remote mode if not auto-detected).")
    c.add_argument("--sglang-url", help="SGLang server URL for the NLA Verbalizer.")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    c.add_argument("--load-in-4bit", action="store_true")
    c.add_argument("--load-in-8bit", action="store_true")
    
    # Sidecar command
    s = sub.add_parser("sidecar", help="Run the NLA Activation Sidecar server.")
    s.add_argument("--model", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--load-in-4bit", action="store_true")

    args = parser.parse_args()
    if args.cmd == "chat":
        try:
            asyncio.run(chat(args))
        except KeyboardInterrupt:
            sys.exit(0)
    elif args.cmd == "sidecar":
        from .sidecar import app as fast_app
        from .providers.local import LocalHFProvider as LProv
        import uvicorn
        import nla_tui.sidecar as sidecar_mod
        
        print(f"Loading model {args.model} for Sidecar...")
        sidecar_mod.provider = LProv(args.model, device=args.device, load_in_4bit=args.load_in_4bit)
        uvicorn.run(fast_app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
