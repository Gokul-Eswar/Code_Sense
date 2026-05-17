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
    """
    Automatically discovers the NLA mapping given a model or model ID.
    Returns the NLAMapping if found, else None.
    """
    model_id = ""
    config = None
    
    if isinstance(model, str):
        model_id = model
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            pass
    elif hasattr(model, "config"):
        config = model.config
        model_id = getattr(config, "_name_or_path", "")
        
    if not config:
        # Fallback to string matching
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

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`registry.py`) acts as a database and lookup service for connecting base models to their 
# corresponding NLA Verbalizer models. It maps model architectures to specific extraction layers.
#
# --- Imports ---
# - `dataclasses.dataclass`: Used to create a clean, structured data container for model mappings.
# - `typing.Optional, Dict, Any`: Used for type hinting variables and function returns.
# - `transformers.AutoConfig`: Used to load model configurations to identify architectures programmatically.
#
# --- Data Structure: NLAMapping ---
# - `base_model_id` (str): The Hugging Face repo ID for the primary model (e.g., Llama-3.3-70B).
# - `nla_model_id` (str): The repo ID for the NLA Verbalizer designed for this model.
# - `extraction_layer` (int): The specific layer index where activations should be intercepted.
# - `d_model` (int): The hidden dimension size of the model (e.g., 4096).
#
# --- Global Variable: _REGISTRY ---
# - `_REGISTRY`: (type: `Dict[str, NLAMapping]`) A hardcoded dictionary of known mappings. 
#   - Key: A human-readable model name.
#   - Value: An `NLAMapping` instance containing technical details.
#
# --- Functions ---
#
# 1. `discover_nla_config(model)`
#    - **Purpose**: Automatically identifies which NLA Verbalizer to use for a given input model.
#    - **Parameters**: `model` (type: `Any`): Can be a model ID string or a loaded Transformers model object.
#    - **Return**: An `NLAMapping` object if a match is found, otherwise `None`.
#    - **Logic Flow**:
#        - **Step 1 (Input Resolution)**: 
#          - If the input is a string, it's treated as a `model_id`. It attempts to load the `AutoConfig`.
#          - If the input is an object, it looks for the `.config` attribute and extracts the `model_id`.
#        - **Step 2 (Fallback Strategy)**: 
#          - If the config cannot be loaded, it performs simple string matching between the `model_id` 
#            and the keys in `_REGISTRY`.
#        - **Step 3 (Architectural Matching)**: 
#          - If the config is available, it extracts `arch` (architecture name), `num_layers`, and `hidden_size`.
#          - It checks specific combinations (e.g., "Qwen2" with 28 layers and 3584 hidden size) to return 
#            the precise mapping. This is more robust than string matching as repo names can change.
#        - **Step 4 (Final Matching)**: 
#          - If architectural matching fails, it does one last pass of case-insensitive string matching.
#        - **Step 5**: Returns `None` if no match is identified.
#
