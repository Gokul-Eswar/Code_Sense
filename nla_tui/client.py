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
