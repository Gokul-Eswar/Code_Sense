from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import AutoConfig

@dataclass
class NLAMapping:
    base_model_id: str
    nla_model_id: str
    extraction_layer: int
    d_model: int

_REGISTRY: Dict[str, NLAMapping] = {
    "Qwen2.5-7B-Instruct": NLAMapping(
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        nla_model_id="kitft/nla-qwen2.5-7b-L20-av",
        extraction_layer=20,
        d_model=3584
    ),
    "Gemma-3-12B-IT": NLAMapping(
        base_model_id="google/gemma-3-12b-it",
        nla_model_id="kitft/nla-gemma3-12b-L32-av",
        extraction_layer=32,
        d_model=3840
    ),
    "Gemma-3-27B-IT": NLAMapping(
        base_model_id="google/gemma-3-27b-it",
        nla_model_id="kitft/nla-gemma3-27b-L41-av",
        extraction_layer=41,
        d_model=5376
    ),
    "Llama-3.3-70B-Instruct": NLAMapping(
        base_model_id="meta-llama/Llama-3.3-70B-Instruct",
        nla_model_id="kitft/Llama-3.3-70B-NLA-L53-av",
        extraction_layer=53,
        d_model=8192
    ),
}

def discover_nla_config(model: Any) -> Optional[NLAMapping]:
    model_id = ""
    config = None
    if isinstance(model, str):
        model_id = model
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except:
            pass
    elif hasattr(model, "config"):
        config = model.config
        model_id = getattr(config, "_name_or_path", "")
    if not config:
        for key, mapping in _REGISTRY.items():
            if key.lower() in model_id.lower():
                return mapping
        return None
    arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else ""
    num_layers = getattr(config, "num_hidden_layers", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    if "Qwen2" in arch and num_layers == 28 and hidden_size == 3584:
        return _REGISTRY["Qwen2.5-7B-Instruct"]
    if "Gemma3" in arch and num_layers == 48 and hidden_size == 3840:
        return _REGISTRY["Gemma-3-12B-IT"]
    if "Gemma3" in arch and num_layers == 62 and hidden_size == 5376:
        return _REGISTRY["Gemma-3-27B-IT"]
    if "Llama" in arch and num_layers == 80 and hidden_size == 8192:
        return _REGISTRY["Llama-3.3-70B-Instruct"]
    for key, mapping in _REGISTRY.items():
        if key.lower() in model_id.lower():
            return mapping
    return None
