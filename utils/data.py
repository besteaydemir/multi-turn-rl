"""Data loading and metadata utilities."""

import pandas as pd
import numpy as np
from datasets import load_dataset


METADATA_CSV_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw/metadata.csv"
_METADATA_CACHE = None


def get_metadata_df():
    """Load metadata CSV with caching."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = pd.read_csv(METADATA_CSV_PATH)
    return _METADATA_CACHE


def sky_direction_to_up_vector(sky_direction):
    """
    Convert sky direction string to camera up vector.
    
    Args:
        sky_direction: Direction string ("Up", "Down", "Left", "Right", or "NA")
    
    Returns:
        3x1 up vector as numpy array
    """
    direction_map = {
        "Up": np.array([0.0, 0.0, 1.0], dtype=float),
        "Down": np.array([0.0, 0.0, -1.0], dtype=float),
        "Left": np.array([-1.0, 0.0, 0.0], dtype=float),
        "Right": np.array([1.0, 0.0, 0.0], dtype=float),
    }
    
    sky_dir_str = str(sky_direction).strip() if sky_direction is not None else "NA"
    up_vector = direction_map.get(sky_dir_str, np.array([0.0, 0.0, 1.0], dtype=float))
    return up_vector


def get_sky_direction_for_scene(scene_id):
    """
    Look up the sky direction for a given scene_id from metadata CSV.
    
    Args:
        scene_id: Scene ID (video_id)
    
    Returns:
        Sky direction string ("Up", "Down", "Left", "Right", or "NA")
    """
    try:
        df = get_metadata_df()
        row = df[df['video_id'] == int(scene_id)]
        if len(row) > 0:
            sky_dir = row.iloc[0]['sky_direction']
            return sky_dir
    except Exception as e:
        print(f"[WARN] Error looking up sky direction for scene {scene_id}: {e}")
    return "NA"


def load_vsi_bench_questions(question_types=None, dataset="arkitscenes"):
    """
    Load VSI-Bench questions filtered by dataset and question type.
    
    Args:
        question_types: List of question types to include, or None for route_planning only.
                       Common types: "route_planning", "object_rel_distance", 
                       "object_rel_direction_easy", "object_rel_direction_medium",
                       "object_rel_direction_hard"
        dataset: Dataset name to filter by (default: "arkitscenes")
    
    Returns:
        List of dicts with: scene_name, question, choices, answer_id, question_type, is_numerical
    """
    print("[INFO] 📥 Loading VSI-Bench dataset...")
    vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
    print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")
    
    # Default to route_planning for backward compatibility
    if question_types is None:
        question_types = ["route_planning"]
    
    # Numerical question types (for MRA evaluation)
    numerical_types = {
        "object_counting",
        "distance_estimation", 
        "size_estimation",
    }
    
    filtered = vsi.filter(
        lambda x: x["dataset"] == dataset
                  and x["question_type"] in question_types
    )
    print(f"[INFO] ✅ Filtered to {len(filtered)} questions")
    
    # Print breakdown by type
    for qt in question_types:
        count = len([x for x in filtered if x['question_type'] == qt])
        if count > 0:
            print(f"[INFO]    - {qt}: {count} questions")
    
    questions = []
    for row in filtered:
        q_type = row.get("question_type", "unknown")
        questions.append({
            "scene_name": row["scene_name"],
            "question": row["question"],
            "choices": row.get("options", []),
            "answer_id": row.get("ground_truth", -1),
            "question_type": q_type,
            "is_numerical": q_type in numerical_types,
        })
    
    return questions


# Multiple choice question types commonly used
MCA_QUESTION_TYPES = [
    "route_planning",
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
]
