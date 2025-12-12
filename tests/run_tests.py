"""
Master test suite runner.

Run all tests before full training to validate:
1. Forward pass alignment
2. Mask correctness
3. Reference model behavior
4. LOO baseline computation
5. Gradient sanity
6. End-to-end integration
7. JSON validity handling
"""

import sys
import pytest
from pathlib import Path


def run_all_tests(verbose=True, stop_on_first_fail=False):
    """
    Run complete test suite.
    
    Args:
        verbose: Print detailed output
        stop_on_first_fail: Stop on first test failure
    
    Returns:
        exit_code: 0 if all tests pass, non-zero otherwise
    """
    # Test files in order of dependency
    test_files = [
        "test_alignment.py",      # Forward pass and token alignment
        "test_advantages.py",      # LOO baseline and gradients
        "test_reference_model.py", # Reference model behavior
        "test_integration.py",     # End-to-end integration
    ]
    
    test_dir = Path(__file__).parent
    
    args = ["-v"] if verbose else []
    if stop_on_first_fail:
        args.append("-x")
    
    # Add markers to skip slow tests by default
    args.extend(["-m", "not slow"])
    
    # Run all test files
    test_paths = [str(test_dir / f) for f in test_files]
    args.extend(test_paths)
    
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"\nTest directory: {test_dir}")
    print(f"Test files: {len(test_files)}")
    print(f"Arguments: {' '.join(args)}\n")
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nYour RL training pipeline is ready!")
        print("\nNext steps:")
        print("1. Collect real episodes with your simulator")
        print("2. Create a config file for your experiment")
        print("3. Run training with: python train.py --config your_config.yaml")
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the failing tests before running full training.")
        print("This ensures your pipeline is working correctly.")
    
    return exit_code


def run_quick_tests():
    """Run only fast tests for quick validation."""
    test_dir = Path(__file__).parent
    
    args = [
        "-v",
        "-m", "not slow",
        str(test_dir / "test_advantages.py"),  # LOO baseline (fast)
        str(test_dir / "test_integration.py::TestJSONValidityHandling"),  # JSON parsing (fast)
    ]
    
    print("Running quick tests (LOO baseline + JSON validation)...\n")
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✓ Quick tests passed")
    else:
        print("\n✗ Quick tests failed")
    
    return exit_code


def run_alignment_tests_only():
    """Run only alignment and forward pass tests."""
    test_dir = Path(__file__).parent
    
    args = [
        "-v", "-s",
        str(test_dir / "test_alignment.py")
    ]
    
    print("Running alignment tests...\n")
    return pytest.main(args)


def run_with_coverage():
    """Run tests with coverage report."""
    test_dir = Path(__file__).parent
    
    args = [
        "--cov=rl_trainer",
        "--cov-report=html",
        "--cov-report=term",
        "-v",
        "-m", "not slow",
        str(test_dir)
    ]
    
    print("Running tests with coverage analysis...\n")
    return pytest.main(args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RL training test suite")
    parser.add_argument(
        "--mode",
        choices=["all", "quick", "alignment", "coverage"],
        default="all",
        help="Which tests to run"
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop on first test failure"
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests (model loading, etc.)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        exit_code = run_quick_tests()
    elif args.mode == "alignment":
        exit_code = run_alignment_tests_only()
    elif args.mode == "coverage":
        exit_code = run_with_coverage()
    else:
        exit_code = run_all_tests(
            verbose=True,
            stop_on_first_fail=args.stop_on_fail
        )
    
    sys.exit(exit_code)
