import httpx
import orjson
import torch
import numpy as np
import yaml
import re
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")
_SCALED_EMBED_MODEL_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text", "t5"})

class NLAClient:
    """
    Client for interacting with the Natural Language Autoencoder (NLA) Verbalizer.
    Supports both remote SGLang execution and robust local metadata verification.
    """
    def __init__(self, nla_model_id: str, sglang_url: Optional[str] = None, device: str = "cpu"):
        self.nla_model_id = nla_model_id
        self.sglang_url = sglang_url.rstrip("/") if sglang_url else None
        self.device = device
        
        # 1. Download metadata and verify tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(nla_model_id, trust_remote_code=True)
        self.cfg = self._load_nla_config()
        
        # 2. Resolve embedding scale and load only embedding weights (efficient)
        self.embed_scale = self._resolve_embed_scale()
        self.embed = self._load_embedding_only().to(device)
        
        self._http = httpx.Client(timeout=httpx.Timeout(120.0))
        self._local_model = None

    def _load_nla_config(self) -> Dict[str, Any]:
        """Load and verify nla_meta.yaml."""
        meta_path = hf_hub_download(repo_id=self.nla_model_id, filename="nla_meta.yaml")
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        
        # Verification logic
        t = meta["tokens"]
        live_inj = self.tokenizer.encode(t["injection_char"], add_special_tokens=False)
        if live_inj != [t["injection_token_id"]]:
            # Some tokenizers might return a list with a BOS token or something similar if not careful
            # But add_special_tokens=False should prevent that.
            if len(live_inj) > 1 and live_inj[-1] == t["injection_token_id"]:
                live_inj = [live_inj[-1]]
            
            if live_inj != [t["injection_token_id"]]:
                print(f"[Warning] Tokenizer mismatch for {t['injection_char']!r}: got {live_inj}, expected [{t['injection_token_id']}]")
            
        return meta

    def _resolve_embed_scale(self) -> float:
        """Resolve model-specific embedding scaling."""
        hf_config = AutoConfig.from_pretrained(self.nla_model_id, trust_remote_code=True)
        text_cfg = getattr(hf_config, "text_config", hf_config)
        model_type = getattr(text_cfg, "model_type", "").lower()
        if model_type in _SCALED_EMBED_MODEL_TYPES:
            return math.sqrt(text_cfg.hidden_size)
        return 1.0

    def _load_embedding_only(self) -> torch.nn.Embedding:
        """Load ONLY the input embedding weight tensor from safetensors."""
        try:
            index_path = hf_hub_download(repo_id=self.nla_model_id, filename="model.safetensors.index.json")
            with open(index_path, "r") as f:
                weight_map = json.load(f)["weight_map"]
            
            # Find the embedding key
            key = next(k for k in weight_map if any(k.endswith(s) for s in _EMBED_KEY_SUFFIXES))
            shard_path = hf_hub_download(repo_id=self.nla_model_id, filename=weight_map[key])
        except Exception:
            # Fallback to single file
            shard_path = hf_hub_download(repo_id=self.nla_model_id, filename="model.safetensors")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            key = next(k for k in f.keys() if any(k.endswith(s) for s in _EMBED_KEY_SUFFIXES))
            weight = f.get_tensor(key).to(torch.bfloat16)

        vocab, d = weight.shape
        embed = torch.nn.Embedding(vocab, d, _weight=weight)
        embed.requires_grad_(False)
        embed.eval()
        return embed

    def _normalize_activation(self, v: torch.Tensor, target_scale: float) -> torch.Tensor:
        """Rescale to target_scale L2-norm."""
        norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return v / (norm_fp32 / target_scale).to(v.dtype)

    def prepare_embeds(self, activation: torch.Tensor) -> np.ndarray:
        """Tokenize -> embed -> scale -> inject."""
        cfg = self.cfg
        content = cfg["prompt_templates"]["av"].format(injection_char=cfg["tokens"]["injection_char"])
        
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True
        )
        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            embeds = (self.embed(ids_t.to(self.device)) * self.embed_scale).float()

        v_scaled = self._normalize_activation(activation.float().view(1, -1), cfg["extraction"]["injection_scale"])
        
        # Injection
        out = embeds.clone()
        inj_id = cfg["tokens"]["injection_token_id"]
        l_id = cfg["tokens"]["injection_left_neighbor_id"]
        r_id = cfg["tokens"]["injection_right_neighbor_id"]
        
        found = False
        for p in range(1, len(input_ids) - 1):
            if input_ids[p] == inj_id and input_ids[p-1] == l_id and input_ids[p+1] == r_id:
                out[0, p] = v_scaled
                found = True
                break
        
        if not found:
            # Fallback for some models where neighbors might be slightly different in TUI vs training
            for p in range(len(input_ids)):
                if input_ids[p] == inj_id:
                    out[0, p] = v_scaled
                    found = True
                    break
            
            if not found:
                raise RuntimeError("Failed to locate precise injection point in prompt.")
            
        return out[0].contiguous().cpu().numpy()

    async def get_thought(self, activation: torch.Tensor, **sampling) -> str:
        """Retrieves natural language explanation."""
        embeds_np = self.prepare_embeds(activation)
        
        if self.sglang_url:
            sp = {"temperature": 1.0, "max_new_tokens": 150}
            sp.update(sampling)
            payload = {"input_embeds": embeds_np, "sampling_params": sp}
            
            # Use the existing sync _http but in an async way or refactor to persistent async client
            # For best performance, we should have self._async_http = httpx.AsyncClient()
            if not hasattr(self, "_async_http") or self._async_http.is_closed:
                self._async_http = httpx.AsyncClient(timeout=30.0)
                
            resp = await self._async_http.post(
                f"{self.sglang_url}/generate",
                content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY),
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()
            text = data[0]["text"] if isinstance(data, list) else data["text"]
        else:
            # Local fallback (Lazy load)
            if self._local_model is None:
                from transformers import AutoModelForCausalLM
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    self.nla_model_id, 
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            embeds = torch.from_numpy(embeds_np).to(self.device, torch.bfloat16).unsqueeze(0)
            with torch.no_grad():
                out = self._local_model.generate(inputs_embeds=embeds, max_new_tokens=150, do_sample=True)
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        m = EXPLANATION_RE.search(text)
        return m.group(1).strip() if m else text.strip()

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`client.py`) defines the `NLAClient` class, which is responsible for turning raw neural activations 
# into natural language "thoughts" by interacting with an NLA Verbalizer model.
#
# --- Imports ---
# - `httpx`: Used for making HTTP requests to remote SGLang servers.
# - `orjson`: A fast JSON library, used here with `OPT_SERIALIZE_NUMPY` to handle NumPy arrays efficiently.
# - `torch`: PyTorch, used for tensor manipulation, normalization, and embedding lookups.
# - `numpy`: Used for high-performance numerical arrays (passed to the API).
# - `yaml`: Used to parse `nla_meta.yaml` configuration files.
# - `re`: Regular expressions, used to extract the explanation from the model's generated text.
# - `math`: Used for square root calculations in embedding scaling.
# - `json`: Standard JSON library for reading index files.
# - `huggingface_hub.hf_hub_download`: Downloads specific files from the Hugging Face Hub.
# - `transformers`: Provides `AutoTokenizer` and `AutoConfig` for interacting with model metadata.
# - `safetensors.safe_open`: Efficiently loads specific tensors from `.safetensors` files without loading the whole model.
#
# --- Constants ---
# - `EXPLANATION_RE`: Regex to find text between `<explanation>` tags.
# - `INJECT_PLACEHOLDER`: String literal used in some prompt templates.
# - `_EMBED_KEY_SUFFIXES`: Common names for embedding weight keys in model state dicts.
# - `_SCALED_EMBED_MODEL_TYPES`: Set of model architectures (like Gemma) that scale embeddings by `sqrt(d_model)`.
#
# --- Class: NLAClient ---
#
# 1. `__init__(self, nla_model_id, sglang_url, device)`
#    - **Purpose**: Initializes the client, downloads metadata, and loads the embedding layer.
#    - **Parameters**: 
#        - `nla_model_id` (str): HF repo ID of the verbalizer.
#        - `sglang_url` (Optional[str]): URL of an SGLang server.
#        - `device` (str): Target device (e.g., "cuda" or "cpu").
#    - **Logic**:
#        - Loads the tokenizer.
#        - Calls `_load_nla_config()` to get verbalizer-specific settings.
#        - Calls `_resolve_embed_scale()` to determine if embeddings need scaling.
#        - Calls `_load_embedding_only()` to load just the necessary weights for prompt construction.
#
# 2. `_load_nla_config(self)`
#    - **Purpose**: Downloads and parses `nla_meta.yaml`, verifying that the tokenizer matches the expected injection token.
#    - **Return**: A dictionary containing NLA metadata.
#
# 3. `_resolve_embed_scale(self)`
#    - **Purpose**: Checks the model config to see if it's a "scaled" architecture (like Gemma).
#    - **Return**: `sqrt(hidden_size)` if scaled, else `1.0`.
#
# 4. `_load_embedding_only(self)`
#    - **Purpose**: Surgical loading of the input embedding weight from HF.
#    - **Logic**: 
#        - Tries to find the embedding shard via the index file.
#        - Uses `safe_open` to extract only the weight tensor.
#        - Returns a `torch.nn.Embedding` object.
#
# 5. `_normalize_activation(self, v, target_scale)`
#    - **Purpose**: Ensures the input activation has the exact L2-norm expected by the verbalizer.
#    - **Logic**: Calculates L2 norm and rescales the vector.
#
# 6. `prepare_embeds(self, activation)`
#    - **Purpose**: Converts a raw activation into a full sequence of input embeddings for the verbalizer.
#    - **Logic**:
#        - Formats the verbalizer prompt (e.g., "What is the model thinking? <char>").
#        - Tokenizes the prompt.
#        - Looks up embeddings for all tokens and scales them.
#        - Normalizes the input `activation` to `injection_scale`.
#        - Replaces the embedding at the "injection point" with the normalized activation vector.
#        - Returns a NumPy array of the final embeddings.
#
# 7. `get_thought(self, activation, **sampling)` (Async)
#    - **Purpose**: The main API method. Takes an activation, returns a string explanation.
#    - **Logic**:
#        - Calls `prepare_embeds` to get the numerical input.
#        - If `sglang_url` is set: Sends a POST request to `/generate` with the `input_embeds`.
#        - If not: Lazily loads the full verbalizer model locally and runs inference.
#        - Uses `EXPLANATION_RE` to clean up the output and return the final thought string.
#
