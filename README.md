# Deterministic vLLM Inference

Achieve reproducible, deterministic responses from vLLM by implementing batch-invariant operations.

## Problem

Standard vLLM inference is non-deterministic because:
- Results depend on batch size (not batch-invariant)
- Server load affects numerical outcomes
- Same prompt + temperature=0 gives different responses

## Solution

Implement batch-invariant kernels that ensure:
- Consistent results regardless of batch size
- Fixed reduction orders in computations
- Deterministic inference across runs

## Usage
```bash
# Basic inference with determinism
python run_inference.py --model MODEL_NAME --deterministic

# Test determinism across multiple runs
python test_determinism.py --trials 10

# Compare standard vs deterministic
python examples/compare_outputs.py