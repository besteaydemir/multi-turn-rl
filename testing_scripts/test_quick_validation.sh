#!/bin/bash
#
# Quick validation test for sequential and video pipelines with numerical questions
# Tests 2 questions from each type (1 MCQ + 1 numerical)
#
# Usage: srun --partition=mcml-dgx-a100-40x8 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --qos=mcml --pty bash test_quick_validation.sh

set -e  # Exit on error

echo "================================================================================"
echo "QUICK VALIDATION TEST - Sequential & Video Pipelines"
echo "Date: $(date)"
echo "================================================================================"

# Activate environment
source /dss/dsshome1/06/di38riq/miniconda3/bin/activate env

# Set vLLM environment variables
export VLLM_USE_MODELSCOPE=False
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Test output directory
TEST_DIR="test_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo ""
echo "Test directory: $TEST_DIR"
echo ""

# Model to test
MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
echo "Using model: $MODEL_ID"
echo ""

#------------------------------------------------------------------------------
# TEST 1: Sequential Pipeline - MCQ Question
#------------------------------------------------------------------------------
echo "================================================================================"
echo "TEST 1: Sequential Pipeline - MCQ (object_rel_distance)"
echo "================================================================================"

python evaluation/sequential.py \
    --backend vllm \
    --dataset arkitscenes \
    --steps 2 \
    --max-questions 1 \
    --test

if [ $? -eq 0 ]; then
    echo "✅ TEST 1 PASSED: Sequential MCQ works"
else
    echo "❌ TEST 1 FAILED: Sequential MCQ failed"
    exit 1
fi

echo ""
sleep 2

#------------------------------------------------------------------------------
# TEST 2: Sequential Pipeline - Numerical Question (Object Counting)
#------------------------------------------------------------------------------
echo "================================================================================"
echo "TEST 2: Sequential Pipeline - Numerical (object_counting)"
echo "================================================================================"

# Create a small test script to load numerical questions
cat > /tmp/test_sequential_numerical.py << 'PYEOF'
import sys
sys.path.insert(0, '/dss/dsshome1/06/di38riq/rl_multi_turn')

from utils import load_vsi_bench_questions, NUMERICAL_QUESTION_TYPES

# Load 1 object_counting question
questions = load_vsi_bench_questions(
    question_types=["object_counting"],
    dataset="arkitscenes"
)

print(f"Loaded {len(questions)} object_counting questions")
if questions:
    print(f"First question: {questions[0]['question'][:60]}...")
    print(f"Is numerical: {questions[0]['is_numerical']}")
    print(f"Scene: {questions[0]['scene_name']}")
PYEOF

python /tmp/test_sequential_numerical.py

if [ $? -eq 0 ]; then
    echo "✅ TEST 2 PASSED: Numerical question loading works"
else
    echo "❌ TEST 2 FAILED: Numerical question loading failed"
    exit 1
fi

echo ""
sleep 2

#------------------------------------------------------------------------------
# TEST 3: Video Pipeline - MCQ Question
#------------------------------------------------------------------------------
echo "================================================================================"
echo "TEST 3: Video Pipeline - MCQ (route_planning)"
echo "================================================================================"

python evaluation/video_baseline.py \
    --backend vllm \
    --dataset arkitscenes \
    --num-frames 4 \
    --max-questions 1 \
    --test

if [ $? -eq 0 ]; then
    echo "✅ TEST 3 PASSED: Video MCQ works"
else
    echo "❌ TEST 3 FAILED: Video MCQ failed"
    exit 1
fi

echo ""
sleep 2

#------------------------------------------------------------------------------
# TEST 4: Video Pipeline - Check Numerical Support
#------------------------------------------------------------------------------
echo "================================================================================"
echo "TEST 4: Video Pipeline - Numerical Question Support Check"
echo "================================================================================"

cat > /tmp/test_video_numerical.py << 'PYEOF'
import sys
sys.path.insert(0, '/dss/dsshome1/06/di38riq/rl_multi_turn')

from evaluation.video_baseline import load_vsi_bench_questions, build_video_prompt, extract_answer

# Test loading with numerical types
questions = load_vsi_bench_questions(
    dataset="arkitscenes",
    include_numerical=True,
    include_temporal=False
)

print(f"Total questions loaded: {len(questions)}")

# Count by type
from collections import Counter
type_counts = Counter([q['question_type'] for q in questions])
print("\nQuestion types:")
for qtype, count in sorted(type_counts.items()):
    print(f"  {qtype}: {count}")

# Test prompt building for numerical type
numerical_q = [q for q in questions if q.get('is_numerical', False)]
if numerical_q:
    sample = numerical_q[0]
    print(f"\nTest numerical prompt for: {sample['question_type']}")
    prompt = build_video_prompt(
        question=sample['question'],
        choices=sample.get('choices', []),
        question_type=sample['question_type'],
        num_frames=4,
        is_numerical=True
    )
    print(f"Prompt length: {len(prompt)} chars")
    print("✅ Numerical prompt generation works")
    
    # Test answer extraction
    test_output = '{"reasoning": "I count 3 objects", "answer": 3}'
    answer = extract_answer(test_output, is_numerical=True)
    print(f"Test extraction: {answer}")
    if answer == "3":
        print("✅ Numerical answer extraction works")
    else:
        print("❌ Numerical answer extraction failed")
        sys.exit(1)
else:
    print("⚠️  No numerical questions found")

PYEOF

python /tmp/test_video_numerical.py

if [ $? -eq 0 ]; then
    echo "✅ TEST 4 PASSED: Video numerical support works"
else
    echo "❌ TEST 4 FAILED: Video numerical support failed"
    exit 1
fi

echo ""
sleep 2

#------------------------------------------------------------------------------
# TEST 5: Answer Extraction for Different Types
#------------------------------------------------------------------------------
echo "================================================================================"
echo "TEST 5: Answer Extraction Validation"
echo "================================================================================"

cat > /tmp/test_answer_extraction.py << 'PYEOF'
import sys
sys.path.insert(0, '/dss/dsshome1/06/di38riq/rl_multi_turn')

from evaluation.video_baseline import extract_answer

test_cases = [
    # MCQ
    ('{"reasoning": "test", "answer": "A"}', False, "A"),
    ('{"reasoning": "test", "answer": "C"}', False, "C"),
    # Numerical integer
    ('{"reasoning": "I count 5", "answer": 5}', True, "5"),
    ('{"reasoning": "3 objects", "answer": 3}', True, "3"),
    # Numerical float
    ('{"reasoning": "1.5 meters", "answer": 1.5}', True, "1.5"),
    ('{"reasoning": "25.3 m2", "answer": 25.3}', True, "25.3"),
]

all_passed = True
for output, is_num, expected in test_cases:
    result = extract_answer(output, is_numerical=is_num)
    if result == expected:
        print(f"✅ {output[:30]:30s} -> {result}")
    else:
        print(f"❌ {output[:30]:30s} -> {result} (expected {expected})")
        all_passed = False

if all_passed:
    print("\n✅ All answer extraction tests passed")
    sys.exit(0)
else:
    print("\n❌ Some answer extraction tests failed")
    sys.exit(1)

PYEOF

python /tmp/test_answer_extraction.py

if [ $? -eq 0 ]; then
    echo "✅ TEST 5 PASSED: Answer extraction validated"
else
    echo "❌ TEST 5 FAILED: Answer extraction issues"
    exit 1
fi

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "✅ ALL TESTS PASSED!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✅ Sequential pipeline works (MCQ)"
echo "  ✅ Sequential numerical question loading works"
echo "  ✅ Video pipeline works (MCQ)"
echo "  ✅ Video numerical support validated"
echo "  ✅ Answer extraction validated (MCQ + numerical)"
echo ""
echo "Ready for large-scale experiments!"
echo ""
echo "Next steps:"
echo "  1. Update submission scripts to include numerical types"
echo "  2. Submit batch jobs for full evaluation"
echo "================================================================================"
