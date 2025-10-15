"""
vLLM Integration and Multi-Model Testing

This replicates the paper's Qwen experiment and extends to other models.
"""

# ============================================================================
# FILE: vllm_deterministic_inference.py
# ============================================================================

"""
Deterministic vLLM inference with batch-invariant operations.

Replicates the paper's experiment:
- Model: Qwen-3-8B (or other models)
- Test: 1000 completions at temperature=0
- Expected: 1 unique completion (vs 18-80 without batch-invariance)
"""


import torch
import hashlib
from collections import Counter
from typing import List, Dict, Optional
import logging

# Try to import vLLM (optional dependency)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not installed. Install with: pip install vllm")

from core import set_batch_invariant_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    """Generate hash of text for comparison."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class DeterministicVLLMEngine:
    """
    Wrapper for vLLM that enables deterministic inference.
    """
    
    def __init__(
        self,
        model_name: str,
        use_batch_invariant: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed")
        
        self.model_name = model_name
        self.use_batch_invariant = use_batch_invariant
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Batch-invariant mode: {use_batch_invariant}")
        
        # Setup determinism
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize vLLM
        with set_batch_invariant_mode(use_batch_invariant):
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                seed=42,
                enforce_eager=True,  # Disable CUDA graphs for determinism
            )
        
        logger.info("✓ Model loaded successfully")
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> List[str]:
        """
        Generate completions for prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            
        Returns:
            List of generated texts
        """
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            seed=42,
            skip_special_tokens=True,
        )
        
        # Generate with batch-invariant operations if enabled
        with set_batch_invariant_mode(self.use_batch_invariant):
            outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text
        completions = [output.outputs[0].text for output in outputs]
        
        return completions


def test_determinism_vllm(
    model_name: str,
    prompt: str = "Tell me about Richard Feynman",
    num_trials: int = 100,
    max_tokens: int = 100,
    use_batch_invariant: bool = True,
) -> Dict:
    """
    Test determinism of vLLM inference.
    
    This replicates the paper's main experiment.
    
    Args:
        model_name: Model to test (e.g., "Qwen/Qwen3-8B")
        prompt: Test prompt
        num_trials: Number of generations to test
        max_tokens: Maximum tokens per generation
        use_batch_invariant: Whether to use batch-invariant ops
        
    Returns:
        Dictionary with results
    """
    logger.info("=" * 70)
    logger.info(f"TESTING DETERMINISM: {model_name}")
    logger.info("=" * 70)
    logger.info(f"Batch-invariant mode: {use_batch_invariant}")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Trials: {num_trials}")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info("")
    
    # Initialize engine
    engine = DeterministicVLLMEngine(
        model_name=model_name,
        use_batch_invariant=use_batch_invariant,
    )
    
    # Run trials
    completions = []
    hashes = []
    
    logger.info(f"Running {num_trials} trials...")
    for i in range(num_trials):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{num_trials}")
        
        # Generate single completion
        completion = engine.generate([prompt], max_tokens=max_tokens)[0]
        completions.append(completion)
        hashes.append(hash_text(completion))
    
    # Analyze results
    unique_hashes = set(hashes)
    hash_counts = Counter(hashes)
    
    results = {
        "model": model_name,
        "batch_invariant": use_batch_invariant,
        "num_trials": num_trials,
        "unique_completions": len(unique_hashes),
        "is_deterministic": (len(unique_hashes) == 1),
        "completions": completions,
        "hash_counts": dict(hash_counts),
    }
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Unique completions: {len(unique_hashes)}/{num_trials}")
    
    if len(unique_hashes) == 1:
        logger.info("✓ DETERMINISTIC: All outputs identical!")
    else:
        logger.info(f"✗ NON-DETERMINISTIC: {len(unique_hashes)} different outputs")
        logger.info("\nMost common outputs:")
        for i, (h, count) in enumerate(sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)[:3], 1):
            logger.info(f"  {i}. Hash {h}: {count}/{num_trials} times")
    
    logger.info(f"\nFirst completion:")
    logger.info(f"  {completions[0][:200]}...")
    
    if len(unique_hashes) > 1:
        # Find first divergent completion
        for i, (comp, h) in enumerate(zip(completions, hashes)):
            if h != hashes[0]:
                logger.info(f"\nFirst different completion (trial {i+1}):")
                logger.info(f"  {comp[:200]}...")
                break
    
    logger.info("=" * 70)
    
    return results

