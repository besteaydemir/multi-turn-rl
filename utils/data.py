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


def load_vsi_bench_questions():
    """
    Load VSI-Bench questions filtered by dataset and question type.
    
    Returns:
        List of dicts with: scene_name, question, choices, answer_id
    """
    print("[INFO] 📥 Loading VSI-Bench dataset...")
    vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
    print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")
    
    filtered = vsi.filter(
        lambda x: x["dataset"] == "arkitscenes"
                  and x["question_type"] == "route_planning"
    )
    print(f"[INFO] ✅ Filtered to {len(filtered)} route_planning questions")
    
    questions = []
    for row in filtered:
        questions.append({
            "scene_name": row["scene_name"],
            "question": row["question"],
            "choices": row.get("options", []),
            "answer_id": row["ground_truth"],
        })
    
    return questions
