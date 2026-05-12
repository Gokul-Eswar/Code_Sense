# Implementation Plan: NLA TUI & Package

## Objective
Build a sleek, lightweight Python package (`nla_tui`) and Terminal UI that visualizes an AI model's internal thoughts in real-time. It must be highly portable (laptops to cloud servers), easily importable as a Python package, and abstract away the complexity of connecting specific models to their corresponding Natural Language Autoencoders.

## Key Constraints & Capabilities
1.  **"Model Independent" Abstraction:** NLA requires specifically trained Verbalizers (AVs) for specific base models (e.g., Qwen, Gemma, Llama). The package will feature an auto-discovery mechanism. When a user passes a model, the tool will inspect its architecture, automatically identify the corresponding NLA checkpoint from the Hugging Face Hub (e.g., `kitft/nla-qwen2.5-7b-L20-av`), and hook into the correct extraction layer without manual configuration.
2.  **Portability & Packaging:** Structured as a standard `pip` installable package (`pip install nla_tui`).
3.  **Two Modes of Operation:**
    *   **As a Library:** A `ThoughtInterceptor` context manager or wrapper that developers can wrap around their existing generation loops.
    *   **As a CLI/TUI:** A standalone `nla-cli` command powered by the `rich` library, providing a beautiful dual-pane terminal interface.

## Implementation Steps

### 1. Package scaffolding
*   Create a new directory/module `nla_tui/`.
*   Setup `pyproject.toml` or `setup.py` for easy installation with dependencies (`rich`, `torch`, `transformers`, `httpx`).

### 2. The Auto-Discovery Registry (`registry.py`)
*   Create a mapping of supported base model architectures to their corresponding NLA HF repo IDs, extraction layers, and injection scales.
*   Implement a factory function that takes an initialized `PreTrainedModel` (or model name) and returns the correct hook configuration.

### 3. The `ThoughtInterceptor` Middleware (`interceptor.py`)
*   Build a PyTorch forward hook that attaches to the target layer defined by the registry.
*   Capture the residual stream vector at the last token position during each generation step.
*   Implement an asynchronous queue to send captured vectors to the NLA Verbalizer backend without blocking the main text generation loop.

### 4. The NLA Backend Client (`client.py`)
*   Implement a lightweight HTTP client (using `httpx`) that communicates with a local or remote `sglang` server hosting the AV model.
*   Include a fallback to run the AV using standard Hugging Face transformers if `sglang` is not available (for pure laptop portability).

### 5. The Terminal UI (`tui.py` & `cli.py`)
*   Use `rich.layout`, `rich.panel`, and `rich.live` to create a sleek dashboard.
*   **Left Pane:** Streams the standard model output token-by-token.
*   **Right Pane (The "Mind"):** Streams the translated thoughts coming back asynchronously from the NLA client.
*   Expose a `nla-cli chat --model Qwen/Qwen2.5-7B-Instruct` command.

## Verification
*   Verify the package installs cleanly via pip.
*   Write a test script verifying the hook correctly captures activations without crashing the generation loop.
*   Run the TUI with a supported model (e.g., Qwen 7B) to ensure the dual-pane UI renders smoothly and asynchronously.
