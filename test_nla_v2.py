import asyncio
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from nla_tui.providers.local import LocalHFProvider

class MockModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = 10
        self.config.architectures = ["Qwen2"]
        self.config._name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        self.model = MagicMock()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(10)])
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        # Simulate streamer behavior
        if "streamer" in kwargs:
            streamer = kwargs["streamer"]
            # Put tokens into streamer
            for token in ["Hello", " world"]:
                streamer.on_finalized_text(token)
                # Manually trigger the hook
                self.model.layers[5](torch.randn(1, 1, 128))
            streamer.end()
        return torch.tensor([[1, 2, 3]])

class MockEncoding(dict):
    def to(self, device):
        return self

@patch("nla_tui.providers.local.AutoTokenizer.from_pretrained")
@patch("nla_tui.providers.local.AutoModelForCausalLM.from_pretrained")
@patch("nla_tui.providers.local.discover_nla_config")
async def test_local_provider(mock_discover, mock_model_load, mock_tokenizer_load):
    print("Testing LocalHFProvider...")
    mock_model_inst = MockModel()
    mock_model_load.return_value = mock_model_inst
    
    mock_tok_inst = MagicMock()
    mock_tokenizer_load.return_value = mock_tok_inst
    mock_tok_inst.apply_chat_template.return_value = "formatted"
    mock_tok_inst.return_value = MockEncoding({"input_ids": torch.tensor([[1]])})
    
    from nla_tui.registry import NLAMapping
    mock_discover.return_value = NLAMapping("base", "nla", 5, 128)
    
    provider = LocalHFProvider("mock/model")
    
    chunks = []
    async for chunk in provider.generate_stream("Hi", []):
        chunks.append(chunk)
        
    assert len(chunks) > 0
    tokens = [c["token"] for c in chunks if c["token"]]
    activations = [c["activation"] for c in chunks if c["activation"] is not None]
    
    print(f"Tokens: {tokens}")
    print(f"Activations captured: {len(activations)}")
    
    assert "Hello" in tokens
    assert len(activations) >= 2
    print("✅ LocalHFProvider test passed!")

if __name__ == "__main__":
    asyncio.run(test_local_provider())