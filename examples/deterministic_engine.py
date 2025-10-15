"""
Deterministic inference engine wrapper.

Wraps model inference with batch-invariant operations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib

from config import DeterministicConfig, DEFAULT_CONFIG
from batch_invariant import batch_invariant_mode


class DeterministicInferenceEngine:
    """
    Inference engine that ensures deterministic outputs.
    """
    
    def __init__(self, config: DeterministicConfig = DEFAULT_CONFIG):
        self.config = config
        self.config.setup_determinism()
        
        self.tokenizer = None
        self.model = None
        self.device = config.device
        
        print(f"Initialized deterministic engine on {self.device}")
    
    def load_model(self, model_name: Optional[str] = None):
        """Load model and tokenizer."""
        model_name = model_name or self.config.model_name
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.config.dtype,
        ).to(self.device)
        
        self.model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        deterministic: bool = True
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            deterministic: Use batch-invariant operations
        
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        max_new_tokens = max_new_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate with batch-invariant operations
        with torch.no_grad():
            if deterministic and self.config.use_batch_invariant:
                with batch_invariant_mode(True):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        do_sample=(temperature > 0),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        deterministic: bool = True
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            deterministic: Use batch-invariant operations
        
        Returns:
            List of generated texts
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        max_new_tokens = max_new_tokens or self.config.max_tokens
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            if deterministic and self.config.use_batch_invariant:
                with batch_invariant_mode(True):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=1.0 if self.config.temperature > 0 else 1.0,
                        do_sample=(self.config.temperature > 0),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(self.config.temperature > 0),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode all outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            text = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            generated_texts.append(text)
        
        return generated_texts


def hash_text(text: str) -> str:
    """Generate hash of text for comparison."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


if __name__ == "__main__":
    # usage
    config = DeterministicConfig(
        model_name="gpt2",
        max_tokens=50,
        temperature=0.0
    )
    
    engine = DeterministicInferenceEngine(config)
    engine.load_model()
    
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")
    
    # Generate multiple times
    print("\nGenerating 3 times with deterministic mode:")
    for i in range(3):
        output = engine.generate(prompt, deterministic=True)
        print(f"{i+1}. {output[:100]}...")