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