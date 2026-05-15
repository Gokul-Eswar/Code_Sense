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
```

## Usage

### Simple Standalone Mode (Laptops)
Run locally using 4-bit quantization. The NLA Verbalizer will also run locally.
```bash
nla-cli chat --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

### High-Performance Mode (Servers)
Run the base model normally and use an external SGLang server for the NLA Verbalizer.
```bash
# Start SGLang server elsewhere:
# python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av --port 30000 --disable-radix-cache

# Run the TUI:
nla-cli chat --model Qwen/Qwen2.5-7B-Instruct --sglang-url http://localhost:30000
```

## Supported Models
- Qwen 2.5 7B
- Gemma 3 (12B, 27B)
- Llama 3.3 70B
