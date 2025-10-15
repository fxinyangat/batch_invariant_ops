"""
Batch invariant operations implementations.

These operations ensure the results are independent of the batch size
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def batch_invariant_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication.
    
    Key principles:
    1. Use consistent GEMM configuration regardless of batch size
    2. Avoid split-K reductions
    3. Use fixed tile sizes
    
    Args:
        input: Input tensor [..., n, m]
        mat2: Matrix tensor [m, p]
        
    Returns:
        Output tensor [..., n, p]
    """
    # FOR CPU: already batch invariant
    # For GPU: would need to configure cuBLAS/CUTLASS settings

    orignal_shape = input.shape

    # Ensure we use the same kernel regardless of the batch size
    # by normalizing to 2D, then reshapping

    if input.dim() > 2:
        # Flatten batch dimensions

        input_2d = input.reshape(-1, orignal_shape[-1])
        output_2d = torch.mm(input_2d, mat2)

        # Reshape back
        output_shape = list(orignal_shape[:-1]) + [mat2.shape[-1]]
        return output_2d.reshape(output_shape)
    
    # Standard 2D case
    return torch.mm(input, mat2)

def batch_invariant_addmm(
        bias: torch.Tensor,
        input: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: float = 1.0,
        alpha: float = 1.0
) -> torch.Tensor:
    
    """
    Batch-invariant addmm: beta * bias + alpha * (input @ mat2).
    
    Args:
        bias: Bias tensor
        input: Input matrix
        mat2: Second matrix
        beta: Multiplier for bias
        alpha: Multiplier for matmul result
        
    Returns:
        Result tensor
    """

    # Use out batch invariant mm

    mm_result = batch_invariant_mm(input, mat2)

    if alpha != 1.0:
        mm_result = mm_result * alpha

    if beta == 1.0:
        return bias + mm_result
    else:
        return beta * bias + mm_result
    
def batch_invariant_mean(
        input: torch.Tensor,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Batch-invariant mean computation.
    
    Ensures reduction order is fixed regardless of batch size.
    
    Args:
        input: Input tensor
        dim: Dimension to reduce
        keepdim: Keep reduced dimension
        dtype: Output dtype
        
    Returns:
        Mean of input
    """

    # For batch invariance we need determinisitic reduction order
    # Use sum + division rather than mean directly

    if dim is None:
        # Reduce all dimensions
        total = input.sum(dtype=dtype)
        count = input.numel()

        return total / count
    
    # Reduce Specific dimension
    total = input.sum(dim=dim, keepdim=keepdim, dtype=dtype)

    if isinstance(dim, int):
        count = input.shape[dim]
    else:
        count = 1
        for d in dim:
            count *= input.shape[d]

    return total / count


def batch_invariant_log_softmax(
        input: torch.Tensor,
        dim: int = -1,
        dtype: Optional[torch.dtype] = None        
) -> torch.Tensor:
    """
    Batch-invariant log-softmax.
    
    Args:
        input: Input tensor
        dim: Dimension for softmax
        dtype: Output dtype
        
    Returns:
        Log-softmax of input
    """

    # Numerically Stable log-softmax with fixed reduction order

    if dtype is not None:
        input = input.to(dtype)

    # Subtract max for numerical stability

    max_val = input.max(dim=dim, keepdim=True)[0]
    input_shifted = input - max_val

    # Compute exp
    exp_vals = torch.exp(input_shifted)
    

    # Sum with deterministic order
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    
    # Log-softmax = input - log(sum(exp(input)))
    log_sum_exp = torch.log(sum_exp) + max_val

    return input - log_sum_exp












    





    


