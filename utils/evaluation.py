"""Evaluation metrics for spatial reasoning tasks."""


def calculate_mra(predicted, ground_truth):
    """
    Calculate Mean Relative Accuracy (MRA) for numerical predictions.
    
    MRA = (1/10) * Σ(θ∈C) 1[|ŷ - y|/y < 1 - θ]
    where C = {0.5, 0.55, 0.60, ..., 0.95}
    
    This metric is commonly used in VSI-Bench for numerical answers.
    
    Args:
        predicted: Model's predicted value (float)
        ground_truth: Ground truth value (float)
    
    Returns:
        MRA score between 0 and 1
    """
    if ground_truth == 0:
        # Avoid division by zero
        return 1.0 if predicted == 0 else 0.0
    
    thresholds = [0.5 + 0.05 * i for i in range(10)]  # [0.5, 0.55, ..., 0.95]
    relative_error = abs(predicted - ground_truth) / abs(ground_truth)
    
    score = 0.0
    for theta in thresholds:
        if relative_error < (1 - theta):
            score += 1.0
    
    return score / 10.0


def evaluate_answer(model_answer, ground_truth, is_numerical=False):
    """
    Evaluate a model answer against ground truth.
    
    Args:
        model_answer: Model's predicted answer (str or float)
        ground_truth: Ground truth answer
        is_numerical: Whether this is a numerical question
    
    Returns:
        Tuple of (is_correct: bool, mra_score: float or None, details: dict)
    """
    details = {
        "model_answer": model_answer,
        "ground_truth": ground_truth,
        "is_numerical": is_numerical
    }
    
    if model_answer == "NO_ANSWER" or model_answer is None:
        return False, None, details
    
    if is_numerical:
        try:
            predicted_value = float(model_answer)
            gt_value = float(ground_truth)
            mra_score = calculate_mra(predicted_value, gt_value)
            is_correct = (mra_score > 0.5)  # Consider correct if MRA > 0.5
            details["predicted_value"] = predicted_value
            details["gt_value"] = gt_value
            details["mra_score"] = mra_score
            return is_correct, mra_score, details
        except (ValueError, TypeError) as e:
            details["parse_error"] = str(e)
            return False, 0.0, details
    else:
        # Multiple choice answer - exact match
        is_correct = (str(model_answer).strip().upper() == str(ground_truth).strip().upper())
        return is_correct, None, details


def compute_accuracy_by_type(results):
    """
    Compute accuracy statistics grouped by question type.
    
    Args:
        results: List of result dicts with 'question_type', 'correct', 'mra_score'
    
    Returns:
        Dict with accuracy stats per question type
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "mra_scores": []})
    
    for r in results:
        if r.get("status") != "COMPLETED":
            continue
        
        q_type = r.get("question_type", "unknown")
        stats[q_type]["total"] += 1
        if r.get("correct"):
            stats[q_type]["correct"] += 1
        if r.get("mra_score") is not None:
            stats[q_type]["mra_scores"].append(r["mra_score"])
    
    # Compute averages
    summary = {}
    for q_type, data in stats.items():
        total = data["total"]
        correct = data["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        summary[q_type] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }
        
        if data["mra_scores"]:
            import statistics
            summary[q_type]["mean_mra"] = statistics.mean(data["mra_scores"])
    
    return summary
