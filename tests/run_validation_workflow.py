"""
Example: Running the test suite before training.

This demonstrates the recommended workflow for validating your RL training pipeline.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """
    Complete validation workflow before training.
    """
    print("=" * 80)
    print("RL TRAINING PIPELINE VALIDATION WORKFLOW")
    print("=" * 80)
    print()
    
    # Step 1: Quick validation (no model loading)
    print("Step 1: Running quick validation tests...")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "tests/run_tests.py", "--mode", "quick"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Quick validation failed!")
        print("Fix the basic tests before proceeding.")
        return False
    
    print("\n✅ Quick validation passed!")
    
    # Step 2: Ask user if they want to run full tests
    print("\n" + "=" * 80)
    print("Step 2: Full test suite (requires model download)")
    print("-" * 80)
    print("\nThe full test suite includes:")
    print("  • Forward pass alignment tests (requires model)")
    print("  • Reference model tests (requires model)")
    print("  • Integration tests (requires model)")
    print()
    
    response = input("Run full test suite? This will download ~5GB model. [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        print("\nRunning full test suite...")
        result = subprocess.run(
            [sys.executable, "tests/run_tests.py", "--include-slow"],
            capture_output=False
        )
        
        if result.returncode != 0:
            print("\n❌ Full test suite failed!")
            print("Please fix the failing tests before training.")
            return False
        
        print("\n✅ Full test suite passed!")
    else:
        print("\nSkipping full test suite.")
        print("You can run it later with: python tests/run_tests.py --include-slow")
    
    # Step 3: Summary and next steps
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE - READY FOR TRAINING")
    print("=" * 80)
    print("\nNext steps:")
    print()
    print("1. Create your training config:")
    print("   python -c 'from config import get_small_scale_config; config = get_small_scale_config(); config.to_yaml(\"my_config.yaml\")'")
    print()
    print("2. Collect episodes with your simulator:")
    print("   python scripts/collect_episodes.py --config my_config.yaml")
    print()
    print("3. Start training:")
    print("   python train.py --config my_config.yaml")
    print()
    print("4. Monitor training:")
    print("   - Logs: tail -f checkpoints/training.log")
    print("   - WandB: https://wandb.ai/<your-project>")
    print("   - TensorBoard: tensorboard --logdir checkpoints/")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
