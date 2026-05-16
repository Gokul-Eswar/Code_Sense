import torch
import torch.nn as nn
import numpy as np
from nla_tui.interceptor import ThoughtInterceptor
from nla_tui.client import NLAClient
from unittest.mock import MagicMock, patch

class MockModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = 10
        self.model = MagicMock()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(10)])

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        return (torch.randn(batch_size, seq_len, self.config.hidden_size),)

def test_interceptor_logic():
    print("Testing ThoughtInterceptor...")
    model = MockModel()
    layer_idx = 5
    interceptor = ThoughtInterceptor(model, layer_idx)
    
    with interceptor:
        model.model.layers[layer_idx](torch.randn(1, 5, 128))
        assert interceptor.queue.qsize() == 1
        activation = interceptor.queue.get()
        assert activation.shape == (128,)
    print("✅ Interceptor logic passed!")

@patch("nla_tui.client.hf_hub_download")
@patch("nla_tui.client.AutoTokenizer.from_pretrained")
@patch("nla_tui.client.AutoConfig.from_pretrained")
@patch("nla_tui.client.safe_open")
def test_client_logic(mock_safe_open, mock_config, mock_tokenizer, mock_download):
    print("Testing NLAClient logic (mocked)...")
    mock_download.return_value = "fake_path"
    mock_tokenizer_inst = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_inst
    mock_tokenizer_inst.encode.return_value = [12345]
    mock_tokenizer_inst.apply_chat_template.return_value = [1, 10, 12345, 11, 2]
    
    mock_config_inst = MagicMock()
    mock_config.return_value = mock_config_inst
    mock_config_inst.hidden_size = 128
    mock_config_inst.model_type = "qwen2"
    
    meta_content = {
        "kind": "nla_model",
        "d_model": 128,
        "tokens": {
            "injection_char": "㊗",
            "injection_token_id": 12345,
            "injection_left_neighbor_id": 10,
            "injection_right_neighbor_id": 11
        },
        "prompt_templates": {"av": "Test {injection_char}"},
        "extraction": {"injection_scale": 1.0}
    }
    
    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load", return_value=meta_content):
            mock_f = MagicMock()
            mock_safe_open.return_value.__enter__.return_value = mock_f
            mock_f.keys.return_value = ["embed_tokens.weight"]
            mock_f.get_tensor.return_value = torch.randn(50000, 128)
            
            client = NLAClient("fake/model")
            activation = torch.randn(128)
            embeds = client.prepare_embeds(activation)
            assert embeds.shape == (5, 128)
            
    print("✅ Client logic passed!")

if __name__ == "__main__":
    try:
        test_interceptor_logic()
        test_client_logic()
        print("\n🎉 All core logic tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Tests failed: {e}")