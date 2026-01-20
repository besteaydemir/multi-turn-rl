"""Camera pose, rotation, and movement utilities."""

import numpy as np
from pathlib import Path


def look_at_camera_pose_center_from_forward(eye, forward=np.array([1.0,0.0,0.0]), up=np.array([0,0,-1])):
    """
    Construct camera-to-world 4x4 matrix at `eye` oriented along `forward` with up `up`.
    
    Args:
        eye: Camera position (3x1)
        forward: Forward direction vector (3x1)
        up: Up direction vector (3x1)
    
    Returns:
        4x4 camera-to-world matrix
    """
    forward = np.asarray(forward, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = forward / forward_norm
    up = np.asarray(up, dtype=float)
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # pick orthogonal up if degenerate
        up_tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(up_tmp, forward)
        right /= np.linalg.norm(right)
    else:
        right = right / right_norm
    true_up = np.cross(forward, right)
    R = np.column_stack((right, true_up, forward))
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3,3] = eye
    return T


def compute_initial_camera_pose(mesh, cam_height=None, up_vector=None):
    """
    Simple and robust initial camera placement.
    
    Args:
        mesh: Open3D TriangleMesh
        cam_height: Optional fixed camera height
        up_vector: Camera up direction
    
    Returns:
        4x4 camera-to-world matrix
    """
    if up_vector is None:
        up_vector = np.array([0.0, 0.0, 1.0], dtype=float)
    
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        raise ValueError("Empty mesh")

    # Get bounding box limits
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    
    # Place camera at center XY
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Use half the Z-range as camera height
    if cam_height is None:
        cam_height = (z_min + z_max) / 2.0
    else:
        cam_height = z_min + cam_height
    
    eye = np.array([center_x, center_y, cam_height], dtype=float)
    
    # Look directly forward (along positive X-axis)
    forward = np.array([1.0, 0.0, 0.0], dtype=float)
    
    pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=up_vector)
    
    print(f"[INFO] 📐 Z-range: [{z_min:.2f}, {z_max:.2f}], cam_height: {cam_height:.2f}")
    print(f"[INFO] 📐 Camera at center: ({center_x:.2f}, {center_y:.2f}, {cam_height:.2f}), looking forward")
    print(f"[INFO] 📐 Up vector: {np.round(up_vector, 2)}")
    
    return pose


def save_matrix(path: Path, mat: np.ndarray, text=True):
    """
    Save a matrix to numpy binary and text formats.
    
    Args:
        path: Path to save (without extension)
        mat: Matrix to save
        text: Whether to also save as text file
    """
    np.save(str(path), mat)
    if text:
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(np.array2string(mat, precision=2, separator=', '))


def parse_rotation_angle(angle_degrees, R_current):
    """
    Apply a rotation angle (in degrees, around z-axis) to the current rotation matrix.
    
    Args:
        angle_degrees: Rotation angle in degrees
        R_current: Current 3x3 rotation matrix
    
    Returns:
        Updated 3x3 rotation matrix
    """
    try:
        angle_rad = float(angle_degrees) * np.pi / 180.0
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        # World-frame rotation: camera turns in world space
        R_new = Rz @ np.array(R_current, dtype=float)
        return R_new
    except Exception as e:
        print(f"[WARN] Failed to apply rotation angle: {e}")
        return np.array(R_current, dtype=float)


def apply_movement_in_camera_frame(R_current, t_current, forward_m, left_m, z_delta_m, bbox_mins=None, bbox_maxs=None):
    """
    Apply movement relative to the camera's current frame, optionally clamped to room bounds.
    
    Args:
        R_current: 3x3 rotation matrix
        t_current: 3x1 translation vector
        forward_m: Forward movement in meters
        left_m: Left movement in meters
        z_delta_m: Vertical movement in meters
        bbox_mins: Optional minimum bounds [x, y, z] for clamping
        bbox_maxs: Optional maximum bounds [x, y, z] for clamping
    
    Returns:
        Updated 3x1 translation vector
    """
    try:
        R = np.array(R_current, dtype=float)
        t = np.array(t_current, dtype=float).reshape(3,)
        
        # Camera frame axes (columns of R are the camera's local axes in world coordinates)
        # R = [right_axis | up_axis | forward_axis]
        right_axis = R[:, 0]    # camera's right direction in world coords
        up_axis = R[:, 1]       # camera's up direction in world coords
        forward_axis = R[:, 2]  # camera's forward direction in world coords
        
        # Movement in world frame
        movement = forward_m * forward_axis + left_m * right_axis
        # Add vertical movement (in world Z)
        movement[2] += z_delta_m
        
        t_new = t + movement
        
        # Clamp to room bounds if provided (with small margin)
        if bbox_mins is not None and bbox_maxs is not None:
            margin = 0.1  # 10cm margin from walls
            t_new[0] = np.clip(t_new[0], bbox_mins[0] + margin, bbox_maxs[0] - margin)
            t_new[1] = np.clip(t_new[1], bbox_mins[1] + margin, bbox_maxs[1] - margin)
            t_new[2] = np.clip(t_new[2], bbox_mins[2] + 0.3, bbox_maxs[2] - 0.3)  # Stay off floor/ceiling
        
        return t_new
    except Exception as e:
        print(f"[WARN] Failed to apply movement: {e}")
        return np.array(t_current, dtype=float)
