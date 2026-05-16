import asyncio
import orjson
import torch
import base64
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from nla_tui.providers.remote import RemoteHTTPProvider

@patch("httpx.AsyncClient.stream")
async def test_remote_provider(mock_stream):
    print("Testing RemoteHTTPProvider with Base64...")
    
    mock_response = MagicMock()
    
    # Mock data with Base64
    act1 = np.array([0.1] * 128, dtype=np.float32)
    act2 = np.array([0.2] * 128, dtype=np.float32)
    
    data1 = {
        "token": "Hello", 
        "activation_b64": base64.b64encode(act1.tobytes()).decode("ascii")
    }
    data2 = {
        "token": " world", 
        "activation_b64": base64.b64encode(act2.tobytes()).decode("ascii")
    }
    
    lines = [
        f"data: {orjson.dumps(data1).decode()}",
        f"data: {orjson.dumps(data2).decode()}",
        "data: [DONE]"
    ]
    
    async def mock_aiter_lines():
        for line in lines:
            yield line
            
    mock_response.aiter_lines = mock_aiter_lines
    mock_stream.return_value.__aenter__.return_value = mock_response
    
    provider = RemoteHTTPProvider("http://localhost:8080")
    
    chunks = []
    async for chunk in provider.generate_stream("Hi", []):
        chunks.append(chunk)
        
    assert len(chunks) == 3
    tokens = [c["token"] for c in chunks if c["token"]]
    activations = [c["activation"] for c in chunks if c["activation"] is not None]
    
    print(f"Tokens: {tokens}")
    print(f"Activations captured: {len(activations)}")
    
    assert tokens == ["Hello", " world"]
    assert len(activations) == 2
    assert activations[0].shape == (128,)
    # Verify values roughly
    assert torch.allclose(activations[0], torch.tensor([0.1] * 128))
    print("✅ RemoteHTTPProvider Base64 test passed!")

if __name__ == "__main__":
    asyncio.run(test_remote_provider())
