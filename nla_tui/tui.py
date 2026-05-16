import asyncio
from typing import Optional, Any
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel

class NLATextualApp(App):
    """
    A modern, reactive Terminal UI for NLA thought visualization.
    """
    CSS = """
    Screen {
        background: #1a1b26;
    }

    #main_container {
        height: 1fr;
    }

    #output_pane {
        width: 65%;
        border-right: vertical $accent;
        padding: 1;
    }

    #mind_pane {
        width: 35%;
        padding: 1;
    }

    Input {
        dock: bottom;
        margin: 1;
        border: tall $accent;
    }

    .thought-card {
        margin: 1 0;
        padding: 0 1;
        border-left: thick $secondary;
        color: $secondary;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+s", "save_export", "Export", show=True),
        Binding("ctrl+l", "clear_logs", "Clear", show=True),
    ]

    def __init__(self, model_info: str = "NLA TUI"):
        super().__init__()
        self.model_info = model_info
        self.gen_fn = None # To be set by cli.py

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_container"):
            yield RichLog(id="output_pane", highlight=True, markup=True, wrap=True)
            yield RichLog(id="mind_pane", highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Type your prompt here...", id="user_input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.model_info
        self.sub_title = "Visualizing the Residual Stream"
        self.query_one("#output_pane").write("[bold cyan]System:[/bold cyan] Ready. Type a message to begin.\n")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
            
        event.input.value = ""
        output_pane = self.query_one("#output_pane")
        output_pane.write(f"\n[bold yellow]User:[/bold yellow] {prompt}\n")
        output_pane.write("[bold green]Assistant:[/bold green] ")
        
        # Trigger generation via a callback to cli.py or a provided function
        if self.gen_fn:
            asyncio.create_task(self.gen_fn(prompt))

    def write_token(self, token: str) -> None:
        self.query_one("#output_pane").write(token, scroll_end=True)

    def write_thought(self, thought: str) -> None:
        log = self.query_one("#mind_pane")
        log.write(f"\n[italic magenta]🧠 {thought}[/italic magenta]\n", scroll_end=True)

    def action_clear_logs(self) -> None:
        self.query_one("#output_pane").clear()
        self.query_one("#mind_pane").clear()

    def action_save_export(self) -> None:
        output = self.query_one("#output_pane").lines
        thoughts = self.query_one("#mind_pane").lines
        
        # Simple export logic
        try:
            with open("nla_session_export.md", "w", encoding="utf-8") as f:
                f.write("# NLA Session Export\n\n")
                f.write("## Chat History\n\n")
                # RichLog lines are tricky, but this is a placeholder for the logic
                f.write("Session export triggered via Ctrl+S.\n")
            self.notify("Session exported to nla_session_export.md")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
