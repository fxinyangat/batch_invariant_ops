
"""
Test determinism across multiple model architectures.

Models to test:
- Qwen/Qwen3-8B (paper's model)
- gpt2 (baseline)
- EleutherAI/gpt-neo-1.3B
- meta-llama/Llama-2-7b-hf (if available)
- google/gemma-2b
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from core import set_batch_invariant_mode, logger


class SimpleModelTester:
    """
    Test determinism with HuggingFace transformers (no vLLM needed).
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("✓ Model loaded")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        use_batch_invariant: bool = True,
    ) -> str:
        """Generate completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            with set_batch_invariant_mode(use_batch_invariant):
                outputs = self