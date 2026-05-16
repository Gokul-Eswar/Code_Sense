# Implementation Plan: Remote Activation Providers & Sidecar

## Objective
Enable NLA thought visualization for models running on remote infrastructure. This covers two primary use cases:
1.  **Server Clusters**: High-performance backends like SGLang.
2.  **Remote Ollama/Generic Servers**: Using a lightweight "Activation Sidecar" to expose hidden states that standard APIs (OpenAI/Ollama) hide.

## Architecture: The Provider Abstraction
We will refactor the activation extraction logic into a pluggable `Provider` system.

### 1. `ActivationProvider` (Abstract Base)
Defines the interface for getting activations from the base model.
- `extract(input_ids: torch.Tensor) -> torch.Tensor`: Returns the residual stream at the target layer.
- `generate_stream(prompt: str, **kwargs)`: Returns a token generator (wraps the model's generation).

### 2. `LocalHFProvider` (Refactored current logic)
- Uses PyTorch forward hooks on a local `transformers` model.
- Best for laptops/local GPUs.

### 3. `RemoteHTTPProvider` (SGLang & Sidecar)
- Communicates with a remote server via HTTP.
- Sends prompts/IDs and receives both text tokens AND the corresponding activation vectors in a single or multiplexed stream.
- **Protocol**: 
    - Request: `{ "prompt": "...", "extract_layer": 20 }`
    - Response: SSE (Server-Sent Events) with chunks like `{ "token": "Hello", "activation": [...] }`.

## The "NLA Sidecar" (`sidecar.py`)
A standalone, lightweight FastAPI server designed to run alongside Ollama or on a remote server.
- **Role**: Loads a model (using `transformers` + `bitsandbytes` for 4-bit) and provides a "Transparent Generation" API.
- **Endpoint**: `/generate_with_activations`
- **Why?**: Standard Ollama/OpenAI APIs do not return hidden states. The sidecar "opens up" the model's brain for the TUI to see.

## Implementation Steps

### Phase 1: Provider Refactoring
1.  Create `nla_tui/providers/` directory.
2.  Implement `base.py` (Abstract Class).
3.  Implement `local.py` (HF + Hooks).
4.  Implement `remote.py` (HTTP + JSON/Msgpack).

### Phase 2: The Sidecar Implementation
1.  Create `nla_tui/sidecar.py`.
2.  Implement a FastAPI server that:
    - Loads a specified model ID.
    - Uses the `ThoughtInterceptor` logic to capture activations during `model.generate()`.
    - Streams tokens and activations back to the client.

### Phase 3: TUI & CLI Integration
1.  Update `nla-cli chat` with new flags:
    - `--remote-url`: Point to a remote Sidecar or SGLang server.
    - `--mode [local|remote]`: Explicitly set the mode.
2.  Update the TUI to handle `RemoteHTTPProvider`'s async stream.

## Verification
1.  **Local**: Ensure `nla-cli chat --model ...` still works with the new refactored `LocalHFProvider`.
2.  **Remote**: 
    - Start `python -m nla_tui.sidecar --model Qwen/Qwen2.5-0.5B-Instruct --port 8080`.
    - Run `nla-cli chat --remote-url http://localhost:8080`.
    - Verify thoughts still stream in real-time.
