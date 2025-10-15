"Core State Management and context manager"

import torch
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Global state

_STATE = {
    "enabled": False,
    "original_ops": {}, # store original Pytorch Operations
    "patched": False,
    "config": {
        "use_fixed_tile_size": True,
        "avoid_split_k": True,
        "log_differences": False
    }
    
 }


def get_batch_invariant_state() -> Dict[str, Any]:
    """Gett current batch invariant state."""

    return _STATE.copy()

def is_batch_invariant_enabled() -> bool:
    """Check if batch invariant is enabled"""
    return _STATE["enabled"]

def _patch_torch_operations():
    """
    Monkey patch Pytorch operations to use batch invariant versions

    This is called once when batch-invariant mode is first enabled.
    """

    if _STATE["patched"]: 
        return
    logger.info("Patching pytocrch operations for batch invariance...")

    # import our batch invariant implementations

    from .operations import(
        batch_invariant_mm,
        batch_invariant_addmm,
        batch_invariant_mean,
        batch_invariant_log_softmax
    )

    #store originals

    _STATE["original_ops"] = {
        "mm": torch.mm,
        "addmm": torch.addmm,
        "mean": torch.mean,
        "log_softmax": torch.nn.functional.log_softmax
    }

    # CREATE Wrapper that checks if mode is enabled

    def make_wrapper(batch_inv_func, original_func):
        def wrapper(*args, **kwargs):
            if _STATE["enabled"]: 
                return batch_inv_func(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)
            
        return wrapper
    
    # patch torch operations
    torch.mm = make_wrapper(batch_invariant_mm, _STATE["original_ops"]["mm"])
    torch.addmm = make_wrapper(batch_invariant_addmm, _STATE['original_ops']['addmm'])
    torch.mean = make_wrapper(batch_invariant_mean, _STATE["original_ops"]["mean"])
    torch.nn.functional.log_softmax = make_wrapper(
        batch_invariant_log_softmax,
        _STATE["original_ops"]["log_softmax"]
        )
    _STATE["patched"] = True

    logger.info("Pytorch Operations patch successfully")

def _unpatch_torch_operations():
    if not _STATE['patched']:
        return
    
    logger.info("Restoring oringinal Pytorch operations...")

    torch.mm = _STATE["original_ops"]["mm"]
    torch.addmm = _STATE["original_ops"]["addmm"]
    torch.mean = _STATE["original_ops"]["mean"]
    torch.nn.functional.log_softmax = _STATE['original_ops']['log_softmax']

    _STATE["patched"] = False
    logger.info("Original torch Operations restored")


@contextmanager
def batch_invariant_mode(
    enabled: bool = True,
    config: Optional[Dict[str,Any]] = None
):
    """
    Context manager to enable/diable batch-invariant mode

    config: optional configuration overides

    Example: 
        >>> with set_batch_invariant_mode(True):
                output = torch.mm(a,b) # Uses batch-invariant version
    """

    #save old state
    old_enabled = _STATE["enabled"]
    old_config = _STATE["config"].copy()

    # Update state
    _STATE["enabled"] = enabled
    if config:
        _STATE["config"].update(config)

    # PAtch operations if enabling first time

    if enabled:
        _patch_torch_operations()
    if enabled:
        logger.info("Batch-invariant mode ENABLED")
    else:
        logger.info("Batch-invariant mode DISABLED")

    try:
        yield
    finally:
        # Restore old state
        _STATE["enabled"] = old_enabled
        _STATE["config"] = old_config
