"""JSON parsing and validation utilities."""

import json
import re
import numpy as np


_JSON_OBJ_RE = re.compile(r"(\{[\s\S]*?\})", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[[\s\S]*?\])", re.DOTALL)


def extract_first_json(text):
    """
    Try to extract a JSON object or array from arbitrary model text.
    
    Args:
        text: String potentially containing JSON
    
    Returns:
        Parsed Python object or None
    """
    # try object first
    m = _JSON_OBJ_RE.search(text)
    if m:
        s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            pass
    # try array (e.g. [ { ... } ])
    m = _JSON_ARRAY_RE.search(text)
    if m:
        s = m.group(1)
        try:
            parsed = json.loads(s)
            # if array, return first element if it's an object
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            return parsed
        except Exception:
            pass
    return None


def parse_qwen_output_and_get_movement(output_text):
    """
    Parse JSON from output_text and extract movement commands.
    
    Args:
        output_text: Model output text
    
    Returns:
        Tuple of (rotation_angle, forward_m, left_m, z_delta_m, reasoning, raw_obj, done)
    """
    obj = extract_first_json(output_text)
    if obj is None:
        return None, None, None, None, None, None, False

    # Reasoning string if present
    reasoning = obj.get("reasoning") if isinstance(obj, dict) else None
    answer = obj.get("answer") if isinstance(obj, dict) else None
    done = obj.get("done", False) if isinstance(obj, dict) else False

    # Try to extract movement parameters
    rotation_angle = None
    forward_m = None
    left_m = None
    z_delta_m = None
    
    if isinstance(obj, dict):
        if "rotation_angle_degrees" in obj:
            rotation_angle = float(obj["rotation_angle_degrees"])
        if "forward_meters" in obj:
            forward_m = float(obj["forward_meters"])
        if "left_meters" in obj:
            left_m = float(obj["left_meters"])
        if "z_delta_meters" in obj:
            z_delta_m = float(obj["z_delta_meters"])

    return rotation_angle, forward_m, left_m, z_delta_m, reasoning, obj, done


def validate_rotation_matrix(R):
    """
    Check if R is 3x3, numeric, orthonormal-ish, and det close to +1.
    If R is close but not quite valid, attempt to fix it via SVD projection.
    
    Args:
        R: Matrix to validate
    
    Returns:
        Tuple of (valid: bool, reason: str, R_corrected: ndarray or None)
    """
    try:
        R = np.array(R, dtype=float)
        if R.shape != (3,3):
            return False, f"shape {R.shape} != (3,3)", None
        
        # Check orthonormality
        RtR = R.T @ R
        err = np.linalg.norm(RtR - np.eye(3))
        det = np.linalg.det(R)
        
        # If orthonormal and det ~1, already valid
        if err < 1e-2 and (0.9 < det < 1.1):
            return True, "", None
        
        # If very close, try to fix via SVD
        if err < 0.05:  # relaxed threshold for attempting fix
            U, S, Vt = np.linalg.svd(R)
            R_fixed = U @ Vt
            # Ensure det = +1 (not -1)
            if np.linalg.det(R_fixed) < 0:
                Vt[-1, :] *= -1
                R_fixed = U @ Vt
            # Verify the corrected matrix is valid
            RtR_fixed = R_fixed.T @ R_fixed
            err_fixed = np.linalg.norm(RtR_fixed - np.eye(3))
            det_fixed = np.linalg.det(R_fixed)
            if err_fixed < 1e-2 and (0.9 < det_fixed < 1.1):
                return True, f"Fixed via SVD (was err={err:.4f}, det={det:.4f})", R_fixed
            else:
                return False, f"SVD correction failed (err_fixed={err_fixed:.4f}, det_fixed={det_fixed:.4f})", None
        
        # Not fixable
        if err > 1e-2:
            return False, f"R^T R error {err:.4f}", None
        if not (0.9 < det < 1.1):
            return False, f"determinant {det:.4f} not ~1", None
        return False, "unknown", None
    except Exception as e:
        return False, f"exception {e}", None


def validate_translation_vector(t):
    """
    Validate a translation vector.
    
    Args:
        t: Vector to validate
    
    Returns:
        Tuple of (valid: bool, reason: str)
    """
    try:
        t = np.array(t, dtype=float).reshape(3,)
        if not np.isfinite(t).all():
            return False, "non-finite"
        return True, ""
    except Exception as e:
        return False, f"exception {e}"
