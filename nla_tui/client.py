import httpx
import orjson
import torch
import numpy as np
import yaml
import re
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open

class NLAClient:
    def __init__(self, nla_model_id: str, sglang_url: Optional[str] = None):
        self.nla_model_id = nla_model_id
        self.sglang_url = sglang_url
        self.config = self._load_meta()
        self.tokenizer = AutoTokenizer.from_pretrained(nla_model_id, trust_remote_code=True)
        self.embed_scale = self._resolve_embed_scale()
        self.weights_path = hf_hub_download(repo_id=nla_model_id, filename="model.safetensors")

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = hf_hub_download(repo_id=self.nla_model_id, filename="nla_meta.yaml")
        with open(meta_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_embed_scale(self) -> float:
        hf_config = AutoConfig.from_pretrained(self.nla_model_id, trust_remote_code=True)
        model_type = getattr(hf_config, "model_type", "").lower()
        if "gemma" in model_type:
            return float(np.sqrt(hf_config.hidden_size))
        return 1.0

    def _get_embedding_weight(self) -> torch.Tensor:
        with safe_open(self.weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "embed_tokens.weight" in key or "wte.weight" in key:
                    return f.get_tensor(key)
        raise ValueError("Could not find embedding weight")

    def prepare_embeds(self, activation: torch.Tensor) -> np.ndarray:
        cfg = self.config
        tokens = cfg["tokens"]
        content = cfg["prompt_templates"]["av"].format(injection_char=tokens["injection_char"])
        input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
        weight = self._get_embedding_weight()
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        embeds = weight[ids_tensor].float() * self.embed_scale
        v_raw = activation.float().view(-1)
        scale = cfg["extraction"]["injection_scale"]
        v_scaled = v_raw / (v_raw.norm().clamp_min(1e-12) / scale)
        inj_id, l_id, r_id = tokens["injection_token_id"], tokens["injection_left_neighbor_id"], tokens["injection_right_neighbor_id"]
        for p in range(1, len(input_ids) - 1):
            if input_ids[p] == inj_id and input_ids[p-1] == l_id and input_ids[p+1] == r_id:
                embeds[p] = v_scaled
                return embeds.numpy()
        raise RuntimeError("No injection point")

    async def get_thought(self, activation: torch.Tensor) -> str:
        if not self.sglang_url: return "[No SGLang URL]"
        payload = {"input_embeds": self.prepare_embeds(activation), "sampling_params": {"temperature": 1.0, "max_new_tokens": 150}}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.sglang_url}/generate", content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))
            resp.raise_for_status()
            text = resp.json()["text"]
        m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()
