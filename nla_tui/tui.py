import asyncio
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

class NLADashboard:
    def __init__(self):
        self.layout = Layout()
        self.out = Text()
        self.mind = Text()
        self.layout.split_row(Layout(name="o"), Layout(name="m"))
        self.layout["o"].update(Panel(self.out, title="Output", border_style="blue"))
        self.layout["m"].update(Panel(self.mind, title="Mind", border_style="magenta"))

    def update(self, token): self.out.append(token)
    def add_thought(self, t): self.mind.append(f"🧠 {t}\n")

async def run_tui(gen_fn, client, interceptor):
    db = NLADashboard()
    with Live(db.layout, refresh_per_second=10) as live:
        async for token in gen_fn():
            db.update(token)
            while not interceptor.queue.empty():
                act = interceptor.queue.get()
                asyncio.create_task(proc_thought(act, client, db))
        await asyncio.sleep(1)

async def proc_thought(act, client, db):
    try: db.add_thought(await client.get_thought(act))
    except Exception as e: db.add_thought(f"[Error: {e}]")
