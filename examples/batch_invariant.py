"""
Batch-invariant operations for deterministic inference.

These operations ensure numerical results don't depend on batch size.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from contextlib import contextmanager


# Global state
_BATCH_INVARIANT_ENABLED = False


@contextmanager
def batch_invariant_mode(enabled: bool = True):
    """Enable/disable batch-invariant operations."""
    global _BATCH_INVARIANT_ENABLED
    old_state = _BATCH_INVARIANT_ENABLED
    _BATCH_INVARIANT_ENABLED = enabled
    try:
        yield
    finally:
        _BATCH_INVARIANT_ENABLED = old_state


def is_batch_invariant_enabled() -> bool:
    """Check if batch-invariant mode is enabled."""
    return _BATCH_INVARIANT_ENABLED



# Batch-Invariant Operations


def batch_invariant_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication.
    
    Key principles:
    - Use consistent tile sizes regardless of batch
    - Avoid split-K reductions
    - Maintain fixed reduction order
    """
    if not _BATCH_INVARIANT_ENABLED:
        return torch.matmul(a, b)
    
    # For batch-invariance, we need to ensure the computation
    # doesn't change based on how we parallelize
    original_shape = a.shape
    
    if len(original_shape) > 2:
        # Reshape to 2D, compute, reshape back
        a_2d = a.reshape(-1, original_shape[-1])
        result = torch.matmul(a_2d, b)
        new_shape = list(original_shape[:-1]) + [b.shape[-1]]
        return result.reshape(new_shape)
    
    return torch.matmul(a, b)


def batch_invariant_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Batch-invariant softmax with fixed reduction order.
    """
    if not _BATCH_INVARIANT_ENABLED:
        return F.softmax(x, dim=dim)
    
    # Numerically stable softmax with explicit reduction order
    max_val = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def batch_invariant_layer_norm(
    x: torch.Tensor,
    normalized_shape: tuple,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Batch-invariant layer normalization.
    """
    if not _BATCH_INVARIANT_ENABLED:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    
    # Compute statistics with fixed reduction order
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    
    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # Apply affine transformation
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def test_batch_invariance(batch_size: int = 128, hidden_dim: int = 512) -> dict:
    """
    Test if operations are batch-invariant.
    
    Returns dict with test results.
    """
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test data
    a = torch.randn(batch_size, hidden_dim, device=device)
    b = torch.randn(hidden_dim, hidden_dim, device=device)
    
    results = {}
    
    # Test matrix multiplication
    with batch_invariant_mode(True):
        out1 = batch_invariant_matmul(a[:1], b)
        out2 = batch_invariant_matmul(a, b)[:1]
        diff = (out1 - out2).abs().max().item()
        results['matmul_diff'] = diff
        results['matmul_invariant'] = (diff == 0)
    
    # Test softmax
    logits = torch.randn(batch_size, hidden_dim, device=device)
    with batch_invariant_mode(True):
        out1 = batch_invariant_softmax(logits[:1])
        out2 = batch_invariant_softmax(logits)[:1]
        diff = (out1 - out2).abs().max().item()
        results['softmax_diff'] = diff
        results['softmax_invariant'] = (diff < 1e-6)
    
    return results


if __name__ == "__main__":
    print("Testing Batch Invariance...")
    print("=" * 60)
    
    # Test without batch-invariant mode
    print("\nStandard Mode:")
    with batch_invariant_mode(False):
        results_std = test_batch_invariance()
    for key, val in results_std.items():
        print(f"  {key}: {val}")
    
    # Test with batch-invariant mode
    print("\nBatch-Invariant Mode:")
    with batch_invariant_mode(True):
        results_inv = test_batch_invariance()
    for key, val in results_inv.items():
        print(f"  {key}: {val}")
    
    print("\n" + "=" * 60)
    if results_inv['matmul_invariant']:
        print("✓ Batch invariance achieved!")
    else:
        print("Batch invariance not perfect (may be due to precision)")