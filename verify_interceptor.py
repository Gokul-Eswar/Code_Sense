import torch
from nla_tui.interceptor import ThoughtInterceptor
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_interceptor():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" # Small model for fast testing
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # We'll use layer 5 for this 0.5B model just to test the hook
    with ThoughtInterceptor(model, 5) as interceptor:
        inputs = tokenizer("Hello world", return_tensors="pt")
        print("Generating...")
        model.generate(**inputs, max_new_tokens=5)
        
        print(f"Captured {interceptor.queue.qsize()} activations.")
        assert interceptor.queue.qsize() > 0
        
        activation = interceptor.queue.get()
        print(f"Activation shape: {activation.shape}")
        assert activation.ndim == 1
        assert activation.shape[0] == model.config.hidden_size

if __name__ == "__main__":
    try:
        test_interceptor()
        print("\n✅ Interceptor test passed!")
    except Exception as e:
        print(f"\n❌ Interceptor test failed: {e}")
