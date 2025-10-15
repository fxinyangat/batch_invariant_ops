"""
Test determinism of inference outputs.
"""

import argparse
from collections import Counter
from typing import List

from config import DeterministicConfig
from deterministic_engine import DeterministicInferenceEngine, hash_text


def test_determinism(
    model_name: str,
    prompt: str,
    num_trials: int = 10,
    max_tokens: int = 100,
    deterministic: bool = True
) -> dict:
    """
    Test if inference is deterministic across multiple runs.
    
    Returns:
        Dict with test results and statistics
    """
    print("=" * 70)
    print(f"Testing Determinism: {'ENABLED' if deterministic else 'DISABLED'}")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Trials: {num_trials}")
    print(f"Max tokens: {max_tokens}")
    print()
    
    # Setup engine
    config = DeterministicConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        use_batch_invariant=deterministic
    )
    
    engine = DeterministicInferenceEngine(config)
    engine.load_model()
    
    # Run multiple trials
    outputs = []
    hashes = []
    
    print(f"Running {num_trials} inference trials...")
    for i in range(num_trials):
        print(f"  Trial {i+1}/{num_trials}", end="\r")
        output = engine.generate(prompt, deterministic=deterministic)
        outputs.append(output)
        hashes.append(hash_text(output))
    
    print()  # New line after progress
    
    # Analyze results
    unique_outputs = len(set(hashes))
    hash_counts = Counter(hashes)
    
    results = {
        "deterministic_mode": deterministic,
        "num_trials": num_trials,
        "unique_outputs": unique_outputs,
        "is_deterministic": (unique_outputs == 1),
        "outputs": outputs,
        "hash_counts": dict(hash_counts)
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Unique outputs: {unique_outputs}/{num_trials}")
    
    if unique_outputs == 1:
        print("DETERMINISTIC: All outputs are identical!")
    else:
        print(f"NON-DETERMINISTIC: Found {unique_outputs} different outputs")
        print("\nOutput frequency:")
        for hash_val, count in sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Hash {hash_val}: {count} times")
    
    print("\nFirst output:")
    print(f"  {outputs[0][:200]}...")
    
    if unique_outputs > 1:
        print("\nSecond unique output:")
        for i, (output, h) in enumerate(zip(outputs, hashes)):
            if h != hashes[0]:
                print(f"  {output[:200]}...")
                break
    
    print("=" * 70)
    
    return results


def compare_modes(
    model_name: str,
    prompt: str,
    num_trials: int = 10,
    max_tokens: int = 100
):
    """
    Compare standard vs deterministic mode.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Standard vs Deterministic Mode")
    print("=" * 70)
    
    # Test standard mode
    print("\n[1/2] Testing STANDARD mode...")
    results_standard = test_determinism(
        model_name=model_name,
        prompt=prompt,
        num_trials=num_trials,
        max_tokens=max_tokens,
        deterministic=False
    )
    
    # Test deterministic mode
    print("\n[2/2] Testing DETERMINISTIC mode...")
    results_deterministic = test_determinism(
        model_name=model_name,
        prompt=prompt,
        num_trials=num_trials,
        max_tokens=max_tokens,
        deterministic=True
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    std_unique = results_standard['unique_outputs']
    det_unique = results_deterministic['unique_outputs']
    
    print(f"Standard mode:      {std_unique}/{num_trials} unique outputs")
    print(f"Deterministic mode: {det_unique}/{num_trials} unique outputs")
    print()
    
    if det_unique == 1:
        print("SUCCESS: Deterministic mode achieved perfect reproducibility!")
    elif det_unique < std_unique:
        print(f"IMPROVEMENT: Reduced unique outputs from {std_unique} to {det_unique}")
    else:
        print("WARNING: No improvement in determinism")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test determinism of LLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Test prompt"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare standard vs deterministic modes"
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable deterministic mode"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_modes(
            model_name=args.model,
            prompt=args.prompt,
            num_trials=args.trials,
            max_tokens=args.max_tokens
        )
    else:
        test_determinism(
            model_name=args.model,
            prompt=args.prompt,
            num_trials=args.trials,
            max_tokens=args.max_tokens,
            deterministic=not args.no_deterministic
        )


if __name__ == "__main__":
    main()