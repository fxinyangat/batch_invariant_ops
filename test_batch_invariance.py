
"""
Test script to verify batch invariance.
"""

import torch
import logging
from core import set_batch_invariant_mode

import torch.nn.functional as F 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mm_batch_invariance():
    """Test matrix multiplication batch invariance."""
    logger.info("=" * 70)
    logger.info("Testing torch.mm Batch Invariance")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Create test data
    torch.manual_seed(42)
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D, device=device).reshape(B, D)
    b = torch.linspace(-100, 100, D*D, device=device).reshape(D, D)
    
    # Test without batch-invariant mode
    logger.info("\n[1] Standard PyTorch:")
    with set_batch_invariant_mode(False):
        out1 = torch.mm(a[:1], b)
        out2 = torch.mm(a, b)[:1]
        diff = (out1 - out2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
        logger.info(f"  Batch-invariant: {diff == 0}")
    
    # Test with batch-invariant mode
    logger.info("\n[2] Batch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        out1 = torch.mm(a[:1], b)
        out2 = torch.mm(a, b)[:1]
        diff = (out1 - out2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
        logger.info(f"  Batch-invariant: {diff == 0}")
    
    logger.info("=" * 70)
    
    return diff == 0


def test_mean_batch_invariance():
    """Test mean computation batch invariance."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing torch.mean Batch Invariance")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    B, D = 1024, 512
    x = torch.randn(B, D, device=device)
    
    logger.info("\n[1] Standard PyTorch:")
    with set_batch_invariant_mode(False):
        mean1 = torch.mean(x[:1], dim=-1)
        mean2 = torch.mean(x, dim=-1)[:1]
        diff = (mean1 - mean2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
    
    logger.info("\n[2] Batch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        mean1 = torch.mean(x[:1], dim=-1)
        mean2 = torch.mean(x, dim=-1)[:1]
        diff = (mean1 - mean2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
    
    logger.info("=" * 70)


def test_log_softmax_batch_invariance():
    """Test log-softmax batch invariance."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing torch.log_softmax Batch Invariance")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    B, D = 512, 1024
    logits = torch.randn(B, D, device=device)
    
    logger.info("\n[1] Standard PyTorch:")
    with set_batch_invariant_mode(False):
        out1 = F.log_softmax(logits[:1], dim=-1)
        out2 = F.log_softmax(logits, dim=-1)[:1]
        diff = (out1 - out2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
    
    logger.info("\n[2] Batch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        out1 = F.log_softmax(logits[:1], dim=-1)
        out2 = F.log_softmax(logits, dim=-1)[:1]
        diff = (out1 - out2).abs().max().item()
        logger.info(f"  Difference: {diff:.6e}")
    
    logger.info("=" * 70)


def main():
    """Run all tests."""
    logger.info("\n🔬 BATCH INVARIANCE TEST SUITE\n")
    
    # Test each operation
    test_mm_batch_invariance()
    test_mean_batch_invariance()
    test_log_softmax_batch_invariance()
    
    logger.info("\n✓ All tests complete!")
    logger.info("\nNote: On CPU, both modes may show identical results")
    logger.info("because CPU PyTorch is already batch-invariant by design.")
    logger.info("The differences appear on GPU with large models.\n")


if __name__ == "__main__":
    main()