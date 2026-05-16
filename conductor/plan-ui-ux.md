# Implementation Plan: Textual UI/UX Modernization

## Objective
Modernize the `nla_tui` interface by migrating from the static `rich.Live` layout to a fully interactive, reactive terminal application using the `Textual` framework. This will provide a drastically improved user experience with robust scrolling, non-blocking input, and a polished aesthetic.

## Key Files & Context
- `pyproject.toml`: Needs `textual` added to dependencies.
- `nla_tui/tui.py`: Will be completely rewritten to define the `Textual` App, Layout, and custom Widgets.
- `nla_tui/cli.py`: The `chat` loop will be refactored to interface with the Textual App instead of driving a manual synchronous prompt loop.
- `nla_tui/client.py` & `interceptor.py`: Remain largely unchanged, but their asynchronous outputs will feed into Textual's reactive properties or message queues.

## Proposed Solution (Textual Architecture)
1.  **Layout**:
    *   **Header**: Displays model name, active device (e.g., CUDA), and SGLang status.
    *   **Body (Horizontal Split)**:
        *   **Left Pane (`RichLog` or `Markdown` widget)**: The main chat history. Scrollable, clean rendering of user prompts and model responses.
        *   **Right Pane (`RichLog` widget)**: The "Mind Stream". Displays the intercepted thoughts asynchronously.
    *   **Footer/Input**:
        *   Persistent `Input` widget at the very bottom.
        *   A `Footer` widget for status indicators ("Thinking...", "Ready") and keybindings (e.g., `Ctrl+S` to Save, `Ctrl+C` to Quit).
2.  **Concurrency Model**: Textual relies on `asyncio`. The model generation (which blocks PyTorch) will run in a separate `threading.Thread` or via `asyncio.to_thread`. Callbacks from the streamer and the `ThoughtInterceptor` queue will use `app.call_from_thread()` to safely update the UI components without blocking the terminal rendering.
3.  **Thought Formatting**: Thoughts will be styled distinctly (e.g., italicized magenta, perhaps bordered) using Textual/Rich markup to differentiate them from standard output.
4.  **Export Feature**: Pressing a bound key (e.g., `Ctrl+S`) will trigger a background task that grabs the content from both the chat log and the thought stream and writes them to a formatted Markdown file (`nla_session_export.md`).

## Implementation Steps
1.  **Dependency Update**: Add `textual>=0.50.0` to `pyproject.toml` and reinstall.
2.  **Design `tui.py` (The App)**:
    *   Create `NLAApp(App)` inheriting from Textual.
    *   Define the `compose()` method with `Header`, `Horizontal` container for the two `RichLog`s, an `Input`, and a `Footer`.
    *   Add integrated CSS for flexbox layout (e.g., Left pane takes 60% width, Right pane takes 40%).
3.  **Refactor `cli.py` (The Controller)**:
    *   Remove the `while True: Prompt.ask()` loop.
    *   Pass the initialized `model`, `tokenizer`, `client`, and `interceptor` to the `NLAApp`.
    *   Call `app.run()`.
4.  **Wire up the Generation Loop**:
    *   Bind the `on_input_submitted` event in Textual to trigger the generation thread.
    *   As tokens yield from the `TextIteratorStreamer`, use `call_from_thread` to write to the Main Chat `RichLog`.
    *   Set up a background asyncio task within the App to constantly poll the `interceptor.queue`, fetch the thought via `client`, and write to the Mind Stream `RichLog`.
5.  **Implement Export**:
    *   Add a standard action `action_save_export` bound to `ctrl+s`.

## Verification & Testing
- Run `nla-cli chat` with a small model.
- Verify that typing while the model is generating does not interrupt the UI rendering.
- Verify both panes can be scrolled independently using the mouse or keyboard.
- Verify `Ctrl+S` successfully writes a Markdown file to disk containing the session.
