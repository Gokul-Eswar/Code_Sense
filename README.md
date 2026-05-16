# NLA TUI

Real-time visualization of AI "thoughts" using Natural Language Autoencoders.

## Key Features
- **Highly Portable:** Runs on everything from a 6GB VRAM laptop to a H100 cloud node.
- **Standalone:** No external server required (optional SGLang support for high throughput).
- **Quantization:** Support for 4-bit and 8-bit loading to fit large models on consumer GPUs.
- **Multi-turn Chat:** Persistent conversation history and thought visualization.

## Install
```bash
pip install -e .
# Ensure you have textual installed
pip install textual
```

## Usage

### Simple Standalone Mode (Local GPU)
Run locally using 4-bit quantization. Both the base model and NLA Verbalizer run on your machine.
```bash
nla-cli chat --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

### Remote / Server Mode (Clusters & Ollama)
If you want to run the base model on a powerful remote server (or alongside your Ollama instance) and run the TUI on your laptop:

1. **On the Server**: Start the NLA Sidecar.
```bash
nla-cli sidecar --model Qwen/Qwen2.5-7B-Instruct --port 8080 --load-in-4bit
```

2. **On your Laptop**: Connect the TUI to the remote server.
```bash
nla-cli chat --remote-url http://<server-ip>:8080 --nla-model kitft/nla-qwen2.5-7b-L20-av
```

*Note: Even in remote mode, the NLA Verbalizer (the "thought translator") runs locally by default to ensure maximum privacy and portability. Use `--sglang-url` if you want to offload the Verbalizer too.*

## Keybindings (in UI)
- **Ctrl+S**: Export the current session to Markdown.
- **Ctrl+L**: Clear the panes.
- **Ctrl+C**: Quit the application.

## Supported Models
- Qwen 2.5 7B
- Gemma 3 (12B, 27B)
- Llama 3.3 70B
