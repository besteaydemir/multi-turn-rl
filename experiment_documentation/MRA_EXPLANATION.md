# Understanding MRA (Mean Relative Accuracy) in VSI-Bench

## What is MRA?

**MRA (Mean Relative Accuracy)** is the evaluation metric used for **numerical answer questions** in VSI-Bench. It provides a more nuanced evaluation than simple exact match, accounting for the difficulty of precise numerical predictions.

## The Formula

```
MRA = (1/10) × Σ(θ∈C) 1[|ŷ - y|/y < 1 - θ]

where:
- C = {0.5, 0.55, 0.60, ..., 0.95} (10 thresholds)
- ŷ = predicted value
- y = ground truth value
- 1[condition] = indicator function (1 if true, 0 if false)
```

## How It Works

MRA evaluates predictions across 10 different tolerance thresholds:

| Threshold (θ) | Required Relative Error | Example (GT=100) |
|---------------|-------------------------|------------------|
| 0.50          | < 50%                   | 50 < pred < 150  |
| 0.55          | < 45%                   | 55 < pred < 145  |
| 0.60          | < 40%                   | 60 < pred < 140  |
| 0.65          | < 35%                   | 65 < pred < 135  |
| 0.70          | < 30%                   | 70 < pred < 130  |
| 0.75          | < 25%                   | 75 < pred < 125  |
| 0.80          | < 20%                   | 80 < pred < 120  |
| 0.85          | < 15%                   | 85 < pred < 115  |
| 0.90          | < 10%                   | 90 < pred < 110  |
| 0.95          | < 5%                    | 95 < pred < 105  |

The MRA score is the **fraction of thresholds passed**, ranging from 0.0 to 1.0.

## Examples

### Example 1: Perfect Prediction
- Ground truth: 100 cm
- Prediction: 100 cm
- Relative error: 0%
- **MRA score: 1.0** (passes all 10 thresholds)

### Example 2: Good Prediction
- Ground truth: 100 cm
- Prediction: 90 cm
- Relative error: 10%
- Passes thresholds: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
- **MRA score: 0.9** (passes 9/10 thresholds)

### Example 3: Moderate Prediction
- Ground truth: 100 cm
- Prediction: 75 cm
- Relative error: 25%
- Passes thresholds: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
- **MRA score: 0.6** (passes 6/10 thresholds)

### Example 4: Poor Prediction
- Ground truth: 100 cm
- Prediction: 40 cm
- Relative error: 60%
- Passes no thresholds
- **MRA score: 0.0**

## Question Types Using MRA

VSI-Bench uses MRA for these numerical question types:

1. **object_counting** - Count objects in the scene
2. **object_size_estimation** - Estimate object dimensions (cm)
3. **room_size_estimation** - Estimate room dimensions (cm)  
4. **object_abs_distance** - Measure absolute distances (cm)

## Why You See a Single Accuracy Score

In the Video Baseline experiment, you see a **single overall accuracy** because:

1. **For MCQ questions** (object_rel_distance, route_planning, etc.):
   - Evaluation: Exact match (A, B, C, D)
   - Score: 1 if correct, 0 if wrong
   
2. **For numerical questions**:
   - Evaluation: MRA score (0.0 to 1.0)
   - Considered "correct" if MRA > 0.5
   - Score: 1 if MRA > 0.5, 0 otherwise

3. **Overall accuracy** = (Number of correct answers) / (Total questions)

## Important Notes

### In Video Baseline Results

Looking at the video baseline code, the `mra_score` column in results.csv is set to `None` for all questions. This means:

- **MCQ questions**: Binary correctness only (no MRA needed)
- **Numerical questions**: The code does exact matching, not MRA calculation
- **Overall accuracy**: Simple percentage of exact matches

**This is a limitation**: The video baseline should calculate proper MRA scores for numerical questions for fair comparison with other methods.

### In Sequential Baseline

The sequential evaluation properly calculates MRA scores for numerical questions:

```python
if is_numerical:
    predicted_value = float(model_answer)
    gt_value = float(ground_truth)
    mra_score = calculate_mra(predicted_value, gt_value)
    is_correct = (mra_score > 0.5)  # Consider correct if MRA > 0.5
```

## Recommended Fix

To get accurate evaluation, the video baseline should be updated to:

1. Calculate MRA for numerical questions
2. Store MRA scores in results.csv
3. Report both:
   - Overall accuracy (binary: MRA > 0.5 for numerical, exact match for MCQ)
   - Average MRA for numerical questions only

## Current Status

**Video Baseline (as of 2026-02-05):**
- Uses exact match for all questions
- Accuracy: 18.67% (4B, 4f) to 26.80% (8B, 32f)
- No MRA scores calculated

**Sequential Baseline (in progress):**
- Properly calculates MRA for numerical questions
- Binary correctness threshold: MRA > 0.5
- Expected to show similar or better performance with multi-turn reasoning

## Summary

The MRA metric allows for more realistic evaluation of numerical predictions by:
- Rewarding approximate answers
- Penalizing proportional errors
- Providing fine-grained scores (not just binary)
- Enabling better comparison across different question difficulties

For a complete evaluation, both exact accuracy and average MRA should be reported separately for numerical question types.
