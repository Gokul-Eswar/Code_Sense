import asyncio
import torch
from typing import AsyncGenerator, Dict, Any, Optional
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from .base import ActivationProvider
from ..interceptor import ThoughtInterceptor
from ..registry import discover_nla_config

class LocalHFProvider(ActivationProvider):
    """
    Provider that runs a HuggingFace model locally and uses forward hooks to extract activations.
    """
    def __init__(self, model_id: str, device: str = "cpu", load_in_4bit: bool = False, load_in_8bit: bool = False):
        self.model_id = model_id
        self.device = device
        
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16 if "cuda" in device and not bnb_config else torch.float32,
            device_map=device, 
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        
        mapping = discover_nla_config(self.model)
        if not mapping:
            raise ValueError(f"Model {model_id} not supported by NLA registry.")
        
        self.mapping = mapping
        self.interceptor = ThoughtInterceptor(self.model, mapping.extraction_layer)

    def get_model_info(self) -> str:
        return f"Local: {self.model_id} ({self.device})"

    async def generate_stream(self, prompt: str, history: list, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        formatted_prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            **ids, 
            streamer=streamer, 
            max_new_tokens=kwargs.get("max_new_tokens", 256), 
            do_sample=True, 
            temperature=kwargs.get("temperature", 0.7)
        )
        
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        with self.interceptor:
            for text in streamer:
                # Get activation if available
                activation = None
                if not self.interceptor.queue.empty():
                    activation = self.interceptor.queue.get()
                
                yield {
                    "token": text,
                    "activation": activation,
                    "done": False
                }
                await asyncio.sleep(0.01)
        
        # Drain remaining activations
        while not self.interceptor.queue.empty():
            yield {
                "token": "",
                "activation": self.interceptor.queue.get(),
                "done": False
            }
            
        yield {"token": "", "activation": None, "done": True}

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`local.py`) implements the `LocalHFProvider`, which is responsible for loading a Hugging Face model, 
# running it on the local hardware, and extracting activations token-by-token using the `ThoughtInterceptor`.
#
# --- Imports ---
# - `asyncio`: Used to integrate the blocking Hugging Face generator with an asynchronous stream.
# - `torch`: PyTorch, used for hardware management and tensor handling.
# - `threading.Thread`: Used to run the model generation in a separate thread so it doesn't block the async loop.
# - `transformers`: Provides the core utilities for loading models (`AutoModelForCausalLM`), tokenizers (`AutoTokenizer`), 
#   streaming text (`TextIteratorStreamer`), and quantization (`BitsAndBytesConfig`).
# - `.base.ActivationProvider`: The parent class defining the interface.
# - `..interceptor.ThoughtInterceptor`: The hook manager that captures hidden states.
# - `..registry.discover_nla_config`: Used to find the correct extraction layer for the loaded model.
#
# --- Class: LocalHFProvider ---
#
# 1. `__init__(self, model_id, device, load_in_4bit, load_in_8bit)`
#    - **Purpose**: Loads the model and tokenizer into memory and sets up the interceptor.
#    - **Parameters**: 
#        - `model_id` (str): Repo ID or local path.
#        - `device` (str): e.g., "cuda" or "cpu".
#        - `load_in_4bit/8bit` (bool): If True, uses `bitsandbytes` to quantize the model for lower memory usage.
#    - **Logic**:
#        - Configures `BitsAndBytesConfig` if quantization is requested.
#        - Loads the tokenizer and model with `trust_remote_code=True`.
#        - Automatically identifies the NLA mapping via `discover_nla_config`.
#        - Initializes the `ThoughtInterceptor` for the model and the mapped layer.
#
# 2. `get_model_info(self)`
#    - **Purpose**: Returns a identification string for the UI.
#
# 3. `generate_stream(self, prompt, history, **kwargs)` (Async)
#    - **Purpose**: Runs inference and yields a stream of tokens and activations.
#    - **Logic Flow**:
#        - **Step 1**: Formats the chat history into a string using the model's chat template.
#        - **Step 2**: Tokenizes the formatted string.
#        - **Step 3**: Sets up a `TextIteratorStreamer`.
#        - **Step 4**: Prepares generation arguments (token limits, temperature, etc.).
#        - **Step 5**: Starts `model.generate()` in a separate `Thread`. This is necessary because `model.generate` is 
#          a blocking synchronous function, but our UI needs to remain responsive.
#        - **Step 6**: Uses the interceptor as a context manager (`with self.interceptor:`).
#        - **Step 7 (Streaming Loop)**: Iterates over the `streamer` (which receives tokens from the background thread).
#            - For each token, it checks `self.interceptor.queue` for a new activation vector.
#            - Yields a dictionary with the token and activation.
#            - Small `asyncio.sleep(0.01)` allows the event loop to process other tasks (like UI updates).
#        - **Step 8 (Cleanup)**: After the main loop, it drains any leftover activations from the queue.
#        - **Step 9**: Yields a `done: True` message.
#
