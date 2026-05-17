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

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`tui.py`) defines the `NLATextualApp` class, which is the visual front-end of the application. 
# It is built using the Textual framework, providing a split-pane view for chat and neural thoughts.
#
# --- Imports ---
# - `asyncio`: Used for asynchronous background tasks.
# - `textual.app.App`: The base class for all Textual applications.
# - `textual.widgets`: UI elements like `Header`, `Footer`, `Input`, and `RichLog` (a scrolling log with markup).
# - `textual.containers`: Layout containers like `Horizontal` and `Vertical`.
# - `rich`: Used for text formatting and styling within the widgets.
#
# --- Class: NLATextualApp ---
#
# 1. `CSS` (Class Attribute)
#    - **Purpose**: Defines the visual styling of the application using a CSS-like syntax.
#    - **Variables**:
#        - `#output_pane`: Takes up 65% of the width, shows the chat history.
#        - `#mind_pane`: Takes up 35% of the width, shows the "thoughts" (activations).
#        - `.thought-card`: Custom styling for thought entries.
#
# 2. `BINDINGS` (Class Attribute)
#    - **Purpose**: Maps keyboard shortcuts to application actions.
#    - **Mappings**:
#        - `ctrl+c` -> `quit`
#        - `ctrl+s` -> `save_export`
#        - `ctrl+l` -> `clear_logs`
#
# 3. `__init__(self, model_info)`
#    - **Purpose**: Initializes the app state.
#    - **Variables**:
#        - `self.model_info` (str): Model name displayed in the header.
#        - `self.gen_fn` (Callable): A callback function (set by `cli.py`) that handles the generation logic.
#
# 4. `compose(self)`
#    - **Purpose**: Defines the widget hierarchy of the UI.
#    - **Logic**: Yields a `Header`, a `Horizontal` container with two `RichLog` widgets, an `Input` field at the bottom, and a `Footer`.
#
# 5. `on_mount(self)`
#    - **Purpose**: Lifecycle event triggered when the app starts.
#    - **Logic**: Sets the window title and writes a "Ready" message to the output pane.
#
# 6. `on_input_submitted(self, event)` (Async)
#    - **Purpose**: Triggered when the user presses Enter in the Input field.
#    - **Logic**:
#        - Extracts and clears the input `prompt`.
#        - Writes the user prompt to the `output_pane`.
#        - Calls `self.gen_fn(prompt)` in a background task to start the model generation.
#
# 7. `write_token(self, token)`
#    - **Purpose**: Appends a generated token to the chat window.
#    - **Logic**: Writes the string to `output_pane` and ensures it scrolls to the end.
#
# 8. `write_thought(self, thought)`
#    - **Purpose**: Appends a verbalized thought to the side window.
#    - **Logic**: Writes the string to `mind_pane` with magenta styling and a brain emoji.
#
# 9. `action_clear_logs(self)`
#    - **Purpose**: Clears both display panes.
#
# 10. `action_save_export(self)`
#    - **Purpose**: Saves the current session to a markdown file.
#    - **Logic**: Opens `nla_session_export.md` and writes a placeholder summary. Displays a notification on success or error.
#
