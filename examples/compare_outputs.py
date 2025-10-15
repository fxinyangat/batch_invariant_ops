"""
Compare outputs between standard and deterministic modes.
"""

from config import DeterministicConfig
from deterministic_engine import DeterministicInferenceEngine, hash_text


def compare_outputs():
    """Compare standard vs deterministic mode outputs."""
    
    config = DeterministicConfig(
        model_name="gpt2",
        max_tokens=50,
        temperature=0.0
    )
    
    engine = DeterministicInferenceEngine(config)
    engine.load_model()
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The secret to happiness is",
    ]
    
    print("=" * 70)
    print("COMPARING STANDARD VS DETERMINISTIC MODE")
    print("=" * 70)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {i}] {prompt}")
        print("-" * 70)
        
        # Standard mode - multiple runs
        print("\nStandard Mode (3 runs):")
        standard_outputs = []
        for j in range(3):
            output = engine.generate(prompt, deterministic=False)
            standard_outputs.append(output)
            print(f"  {j+1}. {output[:80]}...")
        
        # Deterministic mode - multiple runs
        print("\nDeterministic Mode (3 runs):")
        deterministic_outputs = []
        for j in range(3):
            output = engine.generate(prompt, deterministic=True)
            deterministic_outputs.append(output)
            print(f"  {j+1}. {output[:80]}...")
        
        # Check uniqueness
        std_unique = len(set(hash_text(o) for o in standard_outputs))
        det_unique = len(set(hash_text(o) for o in deterministic_outputs))
        
        print(f"\n  Standard: {std_unique}/3 unique outputs")
        print(f"  Deterministic: {det_unique}/3 unique outputs")
        
        if det_unique == 1:
            print("  Yay, Deterministic mode is consistent!")
        else:
            print(" Oops,  Deterministic mode shows variation")
        
        print("=" * 70)


if __name__ == "__main__":
    compare_outputs()