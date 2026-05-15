import httpx
import orjson
import torch
import numpy as np
import yaml
import re
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

class NLAClient:
    """
    Client for interacting with the Natural Language Autoencoder (NLA) Verbalizer.
    Supports both remote SGLang execution and local HuggingFace inference fallback.
    """
    def __init__(self, nla_model_id: str, sglang_url: Optional[str] = None, device: str = "cpu"):
        self.nla_model_id = nla_model_id
        self.sglang_url = sglang_url
        self.device = device
        self.config = self._load_meta()
        self.tokenizer = AutoTokenizer.from_pretrained(nla_model_id, trust_remote_code=True)
        self.embed_scale = self._resolve_embed_scale()
        self.weights_path = hf_hub_download(repo_id=nla_model_id, filename="model.safetensors")
        self._local_model = None

    def _load_local_model(self) -> None:
        """Lazily load the NLA model for local inference if SGLang is not used."""
        if self._local_model is None:
            print(f"Loading NLA model locally: {self.nla_model_id}...")
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.nla_model_id,
                torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )

    def _load_meta(self) -> Dict[str, Any]:
        """Load the nla_meta.yaml sidecar from the Hugging Face hub."""
        meta_path = hf_hub_download(repo_id=self.nla_model_id, filename="nla_meta.yaml")
        with open(meta_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_embed_scale(self) -> float:
        """Resolve model-specific embedding scaling (e.g. Gemma multiplies by sqrt(d_model))."""
        hf_config = AutoConfig.from_pretrained(self.nla_model_id, trust_remote_code=True)
        model_type = getattr(hf_config, "model_type", "").lower()
        if "gemma" in model_type:
            return float(np.sqrt(hf_config.hidden_size))
        return 1.0

    def _get_embedding_weight(self) -> torch.Tensor:
        """Extract the embedding table weight directly from safetensors."""
        with safe_open(self.weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "embed_tokens.weight" in key or "wte.weight" in key:
                    return f.get_tensor(key)
        raise ValueError("Could not find embedding weight in the downloaded safetensors.")

    def prepare_embeds(self, activation: torch.Tensor) -> np.ndarray:
        """
        Injects the activation vector into the fixed prompt embedding.
        
        Args:
            activation: A 1D tensor [hidden_size] representing the residual stream.
            
        Returns:
            A numpy array of shape [seq_len, hidden_size] containing the prompt embeddings.
        """
        cfg = self.config
        tokens = cfg["tokens"]
        
        # 1. Format the template with the injection character
        content = cfg["prompt_templates"]["av"].format(injection_char=tokens["injection_char"])
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        # 2. Get embeddings and apply scale
        weight = self._get_embedding_weight()
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        embeds = weight[ids_tensor].float() * self.embed_scale
        
        # 3. Normalize activation to the expected scale
        v_raw = activation.float().view(-1)
        scale = cfg["extraction"]["injection_scale"]
        v_scaled = v_raw / (v_raw.norm().clamp_min(1e-12) / scale)
        
        # 4. Inject at the marker surrounded by canonical neighbors
        inj_id = tokens["injection_token_id"]
        l_id = tokens["injection_left_neighbor_id"]
        r_id = tokens["injection_right_neighbor_id"]
        
        for p in range(1, len(input_ids) - 1):
            if input_ids[p] == inj_id and input_ids[p-1] == l_id and input_ids[p+1] == r_id:
                embeds[p] = v_scaled
                return embeds.numpy()
                
        raise RuntimeError("Failed to locate the precise injection point with valid neighbors in the prompt.")

    async def get_thought(self, activation: torch.Tensor) -> str:
        """
        Retrieves the natural language explanation for a given activation vector.
        """
        embeds_np = self.prepare_embeds(activation)
        text = ""
        
        try:
            if self.sglang_url:
                payload = {
                    "input_embeds": embeds_np, 
                    "sampling_params": {"temperature": 1.0, "max_new_tokens": 150}
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.sglang_url}/generate", 
                        content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
                    )
                    resp.raise_for_status()
                    text = resp.json()["text"]
            else:
                self._load_local_model()
                embeds = torch.from_numpy(embeds_np).to(self._local_model.device, self._local_model.dtype).unsqueeze(0)
                with torch.no_grad():
                    out = self._local_model.generate(
                        inputs_embeds=embeds, 
                        max_new_tokens=150, 
                        do_sample=True, 
                        temperature=1.0
                    )
                    text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    
        except httpx.ConnectError:
            return "Failed to connect to SGLang server. Ensure it is running or omit --sglang-url for local inference."
        except Exception as e:
            return f"Error analyzing thought: {str(e)}"

        # Attempt to extract text from <explanation> tags
        m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()
