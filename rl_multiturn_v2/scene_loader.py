"""
Scene loading and rendering utilities for RL training.

Integrates with ScanNet, ScanNet++, and ARKitScenes datasets.
Faithfully follows the implementation in evaluation/sequential.py.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import tempfile
import shutil
import uuid

# Import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mesh import find_mesh_file, load_mesh_cached, get_mesh_bounds, clear_mesh_cache
from utils.rendering import render_mesh_from_pose, select_best_initial_view, compute_visibility_score


# ============================================================================
# CONSTANTS (Matching sequential.py)
# ============================================================================

DEFAULT_FX_FY = 300.0
IMAGE_WH = (1024, 768)
CAM_HEIGHT = 1.5  # Camera height from floor (meters)
INITIAL_VIEW_SELECTION_METRIC = "visibility"  # or "laplacian" or "qwen"

# Dataset base directories
DATASET_PATHS = {
    "arkitscenes": "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw",
    "scannet": "/dss/mcmlscratch/06/di38riq/scans",
    "scannetpp": "/dss/mcmlscratch/06/di38riq/data",  # ScanNet++ path
}


@dataclass
class SceneConfig:
    """Configuration for a scene."""
    scene_id: str
    dataset: str
    mesh_path: Path
    bbox_mins: List[float]
    bbox_maxs: List[float]
    initial_pose: Optional[np.ndarray] = None
    
    # Rendering settings
    fx_fy: float = DEFAULT_FX_FY
    image_wh: Tuple[int, int] = IMAGE_WH
    cam_height: float = CAM_HEIGHT


class SceneLoader:
    """
    Loads and manages 3D scenes for RL training.
    
    Supports:
    - ScanNet
    - ScanNet++
    - ARKitScenes
    
    Follows the same loading pattern as evaluation/sequential.py.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
        use_temp_dir: bool = True,
    ):
        """
        Initialize scene loader.
        
        Args:
            output_dir: Directory to save rendered images
            temp_dir: Optional temp directory for renders
            use_temp_dir: If True, create temp directory for renders
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if use_temp_dir:
            self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="rl_renders_"))
        else:
            self.temp_dir = self.output_dir
            
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
        # Cache for loaded scenes
        self._scene_cache: Dict[str, SceneConfig] = {}
        self._mesh_cache: Dict[str, o3d.geometry.TriangleMesh] = {}
        
        # Current scene state
        self.current_scene: Optional[SceneConfig] = None
        self.current_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.current_pose: Optional[np.ndarray] = None
        self.image_counter: int = 0
        
    def _detect_dataset(self, scene_id: str) -> str:
        """
        Detect dataset type from scene_id.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dataset name: "arkitscenes", "scannet", or "scannetpp"
        """
        scene_id_str = str(scene_id)
        
        # ScanNet format: scene0XXX_XX
        if scene_id_str.startswith("scene"):
            return "scannet"
        
        # ScanNet++ format: often alphanumeric like "0a5c013435"
        if len(scene_id_str) == 10 and scene_id_str.isalnum():
            return "scannetpp"
        
        # ARKitScenes format: numeric video IDs like "42897151"
        if scene_id_str.isdigit():
            return "arkitscenes"
        
        # Default to scannetpp if ambiguous
        return "scannetpp"
    
    def _find_mesh_path(self, scene_id: str, dataset: str) -> Optional[Path]:
        """
        Find mesh file path for scene.
        
        Args:
            scene_id: Scene identifier
            dataset: Dataset name
            
        Returns:
            Path to mesh file or None
        """
        base_dir = DATASET_PATHS.get(dataset)
        if not base_dir:
            print(f"[SceneLoader] Unknown dataset: {dataset}")
            return None
        
        if dataset == "scannetpp":
            # ScanNet++ structure: data/scene_id/scans/mesh_aligned_0.05.ply
            mesh_path = Path(base_dir) / scene_id / "scans" / "mesh_aligned_0.05.ply"
            if mesh_path.exists():
                return mesh_path
                
            # Alternative path
            mesh_path = Path(base_dir) / scene_id / "mesh_aligned_0.05.ply"
            if mesh_path.exists():
                return mesh_path
                
            print(f"[SceneLoader] ScanNet++ mesh not found for {scene_id}")
            return None
        else:
            # Use existing find_mesh_file for ARKitScenes and ScanNet
            return find_mesh_file(scene_id, base_dir, dataset)
    
    def load_scene(
        self,
        scene_id: str,
        dataset: Optional[str] = None,
    ) -> SceneConfig:
        """
        Load a scene by ID.
        
        Args:
            scene_id: Scene identifier
            dataset: Optional dataset override (auto-detected if None)
            
        Returns:
            SceneConfig with mesh info
        """
        cache_key = f"{dataset or 'auto'}:{scene_id}"
        
        if cache_key in self._scene_cache:
            config = self._scene_cache[cache_key]
            self.current_scene = config
            self.current_mesh = self._mesh_cache.get(cache_key)
            self.current_pose = config.initial_pose.copy() if config.initial_pose is not None else None
            self.image_counter = 0
            return config
        
        # Detect dataset if not provided
        if dataset is None:
            dataset = self._detect_dataset(scene_id)
            
        print(f"[SceneLoader] Loading scene {scene_id} from {dataset}")
        
        # Find mesh path
        mesh_path = self._find_mesh_path(scene_id, dataset)
        if mesh_path is None:
            raise FileNotFoundError(f"Could not find mesh for scene {scene_id} in {dataset}")
        
        # Load mesh using cached loader
        mesh = load_mesh_cached(mesh_path)
        
        # Get bounding box
        bbox_mins, bbox_maxs = get_mesh_bounds(mesh, percentile_filter=True)
        
        # Create config
        config = SceneConfig(
            scene_id=scene_id,
            dataset=dataset,
            mesh_path=Path(mesh_path),
            bbox_mins=bbox_mins,
            bbox_maxs=bbox_maxs,
        )
        
        # Cache
        self._scene_cache[cache_key] = config
        self._mesh_cache[cache_key] = mesh
        
        # Set current scene
        self.current_scene = config
        self.current_mesh = mesh
        self.current_pose = None
        self.image_counter = 0
        
        return config
    
    def compute_initial_pose(self) -> np.ndarray:
        """
        Compute initial camera pose for current scene.
        
        Places camera at center of scene looking at one of 4 cardinal directions,
        and selects the best view using visibility metric.
        
        Returns:
            4x4 camera-to-world transform matrix
        """
        if self.current_scene is None or self.current_mesh is None:
            raise RuntimeError("No scene loaded")
        
        config = self.current_scene
        mesh = self.current_mesh
        
        # Compute center of scene
        center_x = (config.bbox_mins[0] + config.bbox_maxs[0]) / 2.0
        center_y = (config.bbox_mins[1] + config.bbox_maxs[1]) / 2.0
        cam_height_z = config.bbox_mins[2] + config.cam_height
        eye = np.array([center_x, center_y, cam_height_z], dtype=float)
        
        # Generate 4 candidate views
        view_images = {}
        view_poses = {}
        
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.deg2rad(angle_deg)
            forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
            
            pose = self._look_at_pose(eye, forward)
            view_poses[angle_deg] = pose
            
            # Render candidate view
            img_path = self.temp_dir / f"candidate_{self.current_scene.scene_id}_{angle_deg}.png"
            render_mesh_from_pose(
                mesh, pose, img_path,
                fxfy=config.fx_fy,
                image_wh=config.image_wh
            )
            
            # Load image for scoring
            from PIL import Image
            img_pil = Image.open(img_path)
            img_array = np.array(img_pil).astype(float) / 255.0
            view_images[angle_deg] = img_array
        
        # Select best view
        best_angle, best_score, all_scores = select_best_initial_view(
            view_images, metric=INITIAL_VIEW_SELECTION_METRIC
        )
        
        # Set initial pose
        self.current_pose = view_poses[best_angle].copy()
        config.initial_pose = self.current_pose.copy()
        
        print(f"[SceneLoader] Selected initial view: {best_angle}° (score: {best_score:.3f})")
        
        return self.current_pose
    
    def _look_at_pose(
        self,
        eye: np.ndarray,
        forward: np.ndarray,
        up: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create camera-to-world transform looking at a direction.
        
        Args:
            eye: Camera position (3,)
            forward: Look direction (3,)
            up: Up direction (3,), default is [0, 0, -1]
            
        Returns:
            4x4 camera-to-world matrix
        """
        if up is None:
            up = np.array([0.0, 0.0, -1.0])
        
        # Normalize forward
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Compute right vector
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Recompute up
        up_corrected = np.cross(right, forward)
        up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)
        
        # Build rotation matrix (camera axes)
        R = np.eye(3)
        R[:, 0] = right
        R[:, 1] = up_corrected
        R[:, 2] = -forward  # Camera looks along -Z
        
        # Build 4x4 transform
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = eye
        
        return pose
    
    def render_current_view(self) -> Path:
        """
        Render image from current camera pose.
        
        Returns:
            Path to rendered image
        """
        if self.current_scene is None or self.current_mesh is None or self.current_pose is None:
            raise RuntimeError("No scene loaded or pose not set")
        
        config = self.current_scene
        
        # Create output path
        img_path = self.temp_dir / f"{config.scene_id}_render_{self.image_counter:04d}.png"
        
        # Render
        render_mesh_from_pose(
            self.current_mesh,
            self.current_pose,
            img_path,
            fxfy=config.fx_fy,
            image_wh=config.image_wh
        )
        
        self.image_counter += 1
        
        return img_path
    
    def render_from_action(self, action: 'CameraPose') -> Path:
        """
        Render from a CameraPose action.
        
        Args:
            action: CameraPose from model output
            
        Returns:
            Path to rendered image
        """
        if self.current_scene is None or self.current_mesh is None:
            raise RuntimeError("No scene loaded")
        
        # Convert CameraPose to 4x4 transform matrix
        if action.transform_matrix is not None:
            # Use full transform if available
            self.current_pose = np.array(action.transform_matrix, dtype=float)
        elif action.position is not None and action.rotation is not None:
            # Build from position + rotation
            pose = np.eye(4, dtype=float)
            pose[:3, :3] = np.array(action.rotation, dtype=float)
            pose[:3, 3] = np.array(action.position, dtype=float)
            self.current_pose = pose
        else:
            raise ValueError("CameraPose must have either transform_matrix or (position, rotation)")
        
        return self.render_current_view()
    
    def apply_movement(
        self,
        rotation_angle_degrees: float,
        forward_meters: float,
        left_meters: float,
        z_delta_meters: float = 0.0,
    ) -> Path:
        """
        Apply movement to current pose and render new view.
        
        This follows the movement pattern from sequential.py.
        
        Args:
            rotation_angle_degrees: Rotation angle (-90 to 90)
            forward_meters: Forward movement (-1.0 to 1.0)
            left_meters: Left strafe (-0.5 to 0.5)
            z_delta_meters: Vertical movement (-0.3 to 0.3)
            
        Returns:
            Path to rendered image
        """
        if self.current_pose is None:
            raise RuntimeError("Current pose not set")
        
        config = self.current_scene
        
        # Current rotation and translation
        R_current = self.current_pose[:3, :3]
        t_current = self.current_pose[:3, 3]
        
        # Apply rotation
        R_new = self._apply_rotation(rotation_angle_degrees, R_current)
        
        # Apply translation in camera frame
        t_new = self._apply_movement_in_camera_frame(
            R_new, t_current,
            forward_meters, left_meters, z_delta_meters,
            bbox_mins=config.bbox_mins,
            bbox_maxs=config.bbox_maxs
        )
        
        # Update pose
        self.current_pose[:3, :3] = R_new
        self.current_pose[:3, 3] = t_new
        
        return self.render_current_view()
    
    def _apply_rotation(
        self,
        angle_degrees: float,
        R_current: np.ndarray,
    ) -> np.ndarray:
        """Apply yaw rotation to current rotation matrix."""
        angle_rad = np.deg2rad(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation around Z axis
        R_yaw = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        return R_yaw @ R_current
    
    def _apply_movement_in_camera_frame(
        self,
        R: np.ndarray,
        t: np.ndarray,
        forward_m: float,
        left_m: float,
        z_delta_m: float,
        bbox_mins: List[float],
        bbox_maxs: List[float],
    ) -> np.ndarray:
        """Apply translation in camera frame with bounds clamping."""
        # Camera forward direction in world frame
        forward_dir = R[:, 2]  # Camera Z axis
        forward_dir[2] = 0  # Project to XY plane
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)
        
        # Camera right direction
        right_dir = R[:, 0]
        right_dir[2] = 0
        right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-8)
        
        # Apply movement
        t_new = t.copy()
        t_new[:2] += forward_m * forward_dir[:2]
        t_new[:2] -= left_m * right_dir[:2]  # Left is negative right
        t_new[2] += z_delta_m
        
        # Clamp to bounds with margin
        margin = 0.3
        t_new[0] = np.clip(t_new[0], bbox_mins[0] + margin, bbox_maxs[0] - margin)
        t_new[1] = np.clip(t_new[1], bbox_mins[1] + margin, bbox_maxs[1] - margin)
        t_new[2] = np.clip(t_new[2], bbox_mins[2] + 0.5, bbox_maxs[2] - 0.3)
        
        return t_new
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get current scene information."""
        if self.current_scene is None:
            return {}
        
        config = self.current_scene
        return {
            "scene_id": config.scene_id,
            "dataset": config.dataset,
            "mesh_path": str(config.mesh_path),
            "bbox_mins": config.bbox_mins,
            "bbox_maxs": config.bbox_maxs,
            "image_count": self.image_counter,
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists() and "rl_renders_" in str(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"[SceneLoader] Cleaned up temp directory: {self.temp_dir}")
    
    def __del__(self):
        """Destructor to clean up."""
        try:
            self.cleanup()
        except:
            pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_render_fn(scene_loader: SceneLoader):
    """
    Create a render function compatible with VLLMRolloutEngine.
    
    Args:
        scene_loader: SceneLoader instance
        
    Returns:
        Callable that takes CameraPose and returns image path
    """
    def render_fn(camera_pose) -> str:
        """Render from camera pose action."""
        img_path = scene_loader.render_from_action(camera_pose)
        return str(img_path)
    
    return render_fn


def create_movement_render_fn(scene_loader: SceneLoader):
    """
    Create a render function that handles movement commands.
    
    Args:
        scene_loader: SceneLoader instance
        
    Returns:
        Callable that takes movement dict and returns image path
    """
    def render_fn(movement: Dict[str, float]) -> str:
        """Render after applying movement."""
        img_path = scene_loader.apply_movement(
            rotation_angle_degrees=movement.get("rotation_angle_degrees", 0),
            forward_meters=movement.get("forward_meters", 0),
            left_meters=movement.get("left_meters", 0),
            z_delta_meters=movement.get("z_delta_meters", 0),
        )
        return str(img_path)
    
    return render_fn


# ============================================================================
# VSI-BENCH DATA LOADER
# ============================================================================

def load_vsi_bench_questions(
    dataset: str = "combined",
    question_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load VSI-Bench questions.
    
    Args:
        dataset: "arkitscenes", "scannet", "scannetpp", or "combined"
        question_types: Optional filter for question types
        
    Returns:
        List of question dicts
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] datasets package not installed, returning empty list")
        return []
    
    # Multiple choice question types
    MCA_QUESTION_TYPES = [
        "object_rel_direction_easy",
        "object_rel_direction_medium", 
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning"
    ]
    
    if question_types is None:
        question_types = MCA_QUESTION_TYPES
    
    questions = []
    
    def load_questions_for_dataset(ds_name: str):
        try:
            ds = load_dataset("nyu-visionx/VSI-Bench", split="test")
            
            filtered = []
            for item in ds:
                # Filter by dataset
                item_dataset = item.get("dataset", "").lower()
                if ds_name == "arkitscenes" and "arkit" not in item_dataset:
                    continue
                if ds_name == "scannet" and "scannet" not in item_dataset:
                    continue
                if ds_name == "scannetpp" and "scannet++" not in item_dataset:
                    continue
                
                # Filter by question type
                q_type = item.get("question_type", "")
                if q_type not in question_types:
                    continue
                
                filtered.append({
                    "question": item["question"],
                    "choices": item.get("choices", []),
                    "scene_id": item["scene_id"],
                    "ground_truth": item.get("ground_truth", item.get("answer")),
                    "question_type": q_type,
                    "question_id": item.get("question_id", len(filtered)),
                    "dataset": ds_name,
                })
            
            return filtered
        except Exception as e:
            print(f"[WARN] Failed to load VSI-Bench for {ds_name}: {e}")
            return []
    
    if dataset == "combined":
        questions.extend(load_questions_for_dataset("arkitscenes"))
        questions.extend(load_questions_for_dataset("scannet"))
        questions.extend(load_questions_for_dataset("scannetpp"))
    else:
        questions = load_questions_for_dataset(dataset)
    
    print(f"[SceneLoader] Loaded {len(questions)} questions from {dataset}")
    
    return questions
