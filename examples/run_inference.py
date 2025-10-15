"""
Run inference with deterministic engine.
"""

import argparse
from config import DeterministicConfig
from deterministic_engine import DeterministicInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run deterministic LLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic mode (batch-invariant operations)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to generate"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = DeterministicConfig(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_batch_invariant=args.deterministic
    )
    
    # Initialize engine
    print("Initializing deterministic inference engine...")
    engine = DeterministicInferenceEngine(config)
    engine.load_model()
    
    # Run inference
    print("\n" + "=" * 70)
    print(f"Prompt: {args.prompt}")
    print(f"Mode: {'DETERMINISTIC' if args.deterministic else 'STANDARD'}")
    print("=" * 70)
    
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n[Generation {i+1}/{args.repeat}]")
        
        output = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            deterministic=args.deterministic
        )
        
        print(output)
        print("-" * 70)


if __name__ == "__main__":
    main()