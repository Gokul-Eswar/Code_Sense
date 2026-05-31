import asyncio
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from nla_tui.providers.local import LocalHFProvider
from nla_tui.tui import NLATextualApp
import os

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

async def test_tui_export():
    print("Testing TUI export action (Ctrl+S)...")
    app = NLATextualApp(model_info="Test Model")
    
    # We run the app in headless/test mode using Textual's run_test() context manager
    async with app.run_test() as pilot:
        # Write dummy lines directly into the RichLogs
        output_pane = app.query_one("#output_pane")
        mind_pane = app.query_one("#mind_pane")
        
        output_pane.write("User: hello")
        output_pane.write("Assistant: hi")
        mind_pane.write("🧠 thought 1")
        
        # We need a small yield to let the writes flush
        await asyncio.sleep(0.1)
        
        # Trigger Ctrl+S export action
        await pilot.press("ctrl+s")
        
        # Verify file was written
        assert os.path.exists("nla_session_export.md")
        with open("nla_session_export.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"Exported content:\n{content}")
        assert "User: hello" in content
        assert "Assistant: hi" in content
        assert "🧠 thought 1" in content
        
        # Clean up
        os.remove("nla_session_export.md")
        
    print("✅ TUI export test passed!")

async def main():
    await test_local_provider()
    await test_tui_export()

if __name__ == "__main__":
    asyncio.run(main())