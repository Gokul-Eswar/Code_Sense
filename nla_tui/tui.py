import asyncio
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align

class NLADashboard:
    """
    A rich-based Terminal UI for displaying model outputs and internal thoughts side-by-side.
    """
    def __init__(self):
        self.layout = Layout()
        self.out = Text()
        self.mind = Text()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        self.layout["main"].split_row(
            Layout(name="output", ratio=2),
            Layout(name="mind", ratio=1)
        )
        self.layout["header"].update(
            Panel(Align.center("[bold cyan]NLA TUI[/bold cyan] - Visualization of Internal States"), style="cyan")
        )
        self._update_panels()

    def _update_panels(self):
        self.layout["main"]["output"].update(
            Panel(self.out, title="[bold blue]Model Output", border_style="blue", padding=(1, 2))
        )
        self.layout["main"]["mind"].update(
            Panel(self.mind, title="[bold magenta]The Mind", border_style="magenta", padding=(1, 2))
        )

    def update(self, token: str): 
        self.out.append(token)
        self._update_panels()

    def add_thought(self, t: str): 
        self.mind.append(f"🧠 {t}\n\n")
        self._update_panels()

async def run_tui(gen_fn, client, interceptor, dashboard=None):
    if dashboard is None:
        dashboard = NLADashboard()
    
    # Use screen=False so we don't clear the terminal when it exits
    with Live(dashboard.layout, refresh_per_second=15, screen=False) as live:
        async for token in gen_fn():
            dashboard.update(token)
            while not interceptor.queue.empty():
                act = interceptor.queue.get()
                asyncio.create_task(proc_thought(act, client, dashboard))
                
        # Drain remaining thoughts in the queue
        await asyncio.sleep(0.5)
        while not interceptor.queue.empty():
             act = interceptor.queue.get()
             await proc_thought(act, client, dashboard)
             
    return dashboard

async def proc_thought(act, client, db):
    try: 
        thought = await client.get_thought(act)
        db.add_thought(thought)
    except Exception as e: 
        db.add_thought(f"[red]Error:[/red] {e}")
