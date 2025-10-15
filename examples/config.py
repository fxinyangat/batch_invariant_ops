
"""Configuration for deterministic inference."""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeterministicConfig:
    """Configuration for deterministic inference."""
    
    # Determinism settings
    seed: int = 42
    use_batch_invariant: bool = True
    
    # Model settings
    model_name: str = "gpt2"
    max_tokens: int = 100
    temperature: float = 0.0
    
    # Performance settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # Testing settings
    num_trials: int = 10
    
    def setup_determinism(self):
        """Configure PyTorch for deterministic execution."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Enable deterministic algorithms where possible
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"Warning: Could not enable all deterministic algorithms: {e}")


# Default configuration
DEFAULT_CONFIG = DeterministicConfig()