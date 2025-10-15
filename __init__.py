"""
Battch-Invariant Operations for Determinsitic LLM Inference

Usage:
    from batch_invariant_ops import set_batch_invariant_mode

    with set_batch_invariant_mode(True):
        output = model(input_tensor)

"""


from .core import(
    set_batch_invariant_mode,
    is_batch_invariant_enabled,
    get_batch_invariant_state
)

from .operations import (
    batch_invariant_mm,
    batch_invariant_addmm,
    batch_invariant_mean,
    batch_invariant_log_softmax,
    
)


__version__ = "1.0.0"
__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_enabled",
    "get_batch_invariant_state",
    "batch_invariant_mm",
    "batch_invariant_addmm",
    "batch_invariant_mean",
    "batch_invariant_log_softmax"
]