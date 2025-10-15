# test_without_vllm.py
"""
Test determinism using HuggingFace Transformers (no vLLM needed).
This works on CPU and doesn't require vLLM.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from core import set_batch_invariant_mode
import hashlib
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_model_determinism(
    model_name: str = "gpt2",
    prompt: str = "The future of AI is",
    num_trials: int = 10,
    max_new_tokens: int = 50,
):
    """Test determinism with any HuggingFace model."""
    
    logger.info(f"Testing {model_name}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Test with batch-invariant mode
    logger.info("\nTesting with batch-invariant mode:")
    completions = []
    
    with set_batch_invariant_mode(True):
        for i in range(num_trials):
            torch.manual_seed(42)  # Reset seed each time
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completions.append(text)
            
            if i == 0:
                logger.info(f"First output: {text[:100]}...")
    
    # Check uniqueness
    hashes = [hash_text(c) for c in completions]
    unique = len(set(hashes))
    
    logger.info(f"\nResults: {unique}/{num_trials} unique outputs")
    
    if unique == 1:
        logger.info("✓ DETERMINISTIC: All outputs identical!")
    else:
        logger.info(f"⚠ {unique} different outputs found")
        counts = Counter(hashes)
        logger.info("Hash distribution:")
        for h, count in counts.most_common():
            logger.info(f"  {h}: {count} times")


if __name__ == "__main__":
    # Test with different models
    models = [
        "gpt2",
        "distilgpt2",
        # "Qwen/Qwen2.5-0.5B-Instruct",  # If you have enough memory
    ]
    
    for model_name in models:
        print("\n" + "=" * 70)
        test_model_determinism(model_name)
        print("=" * 70)