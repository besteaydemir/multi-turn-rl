"""Mesh loading and caching utilities."""

from pathlib import Path
import open3d as o3d


# Default mesh base directory
DEFAULT_MESH_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"

# Mesh cache to avoid reloading the same mesh multiple times
_mesh_cache = {}


def find_mesh_file(scene_id, mesh_base_dir=DEFAULT_MESH_BASE_DIR):
    """
    Find a mesh file for the given scene_id.
    
    Searches in both Validation and Training splits.
    
    Args:
        scene_id: Scene ID (video_id string)
        mesh_base_dir: Base directory for mesh files
    
    Returns:
        Path to mesh file, or None if not found
    """
    video_id = str(scene_id)
    for split in ["Validation", "Training"]:
        mesh_path = Path(mesh_base_dir) / split / video_id / f"{video_id}_3dod_mesh.ply"
        if mesh_path.exists():
            return mesh_path
    print(f"[WARN] Mesh file not found for scene {scene_id} in {mesh_base_dir}")
    return None


def load_mesh_cached(mesh_path, max_cache_size=5):
    """
    Load mesh with caching to avoid redundant disk reads.
    
    Args:
        mesh_path: Path to the mesh file
        max_cache_size: Maximum number of meshes to keep in cache
    
    Returns:
        Open3D TriangleMesh object
    
    Raises:
        RuntimeError: If mesh file is empty
    """
    mesh_path_str = str(mesh_path)
    if mesh_path_str not in _mesh_cache:
        print(f"[INFO] 📂 Loading mesh (caching): {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path_str)
        if mesh.is_empty():
            raise RuntimeError(f"Loaded mesh is empty: {mesh_path}")
        _mesh_cache[mesh_path_str] = mesh
        # Limit cache size to avoid memory issues
        if len(_mesh_cache) > max_cache_size:
            oldest_key = next(iter(_mesh_cache))
            del _mesh_cache[oldest_key]
            print(f"[INFO] 🗑️  Evicted oldest mesh from cache")
    else:
        print(f"[INFO] ✅ Using cached mesh: {mesh_path}")
    return _mesh_cache[mesh_path_str]


def clear_mesh_cache():
    """Clear all cached meshes."""
    global _mesh_cache
    _mesh_cache.clear()
    print("[INFO] 🗑️  Mesh cache cleared")


def get_mesh_bounds(mesh, percentile_filter=True):
    """
    Get bounding box of mesh vertices.
    
    Args:
        mesh: Open3D TriangleMesh
        percentile_filter: If True, use 2nd-98th percentile to filter outliers
    
    Returns:
        Tuple of (bbox_mins, bbox_maxs) as lists
    """
    import numpy as np
    
    vertices = np.asarray(mesh.vertices)
    
    if percentile_filter:
        # Use percentile-based bounding box to filter outliers
        bbox_mins = np.percentile(vertices, 2, axis=0).tolist()
        bbox_maxs = np.percentile(vertices, 98, axis=0).tolist()
    else:
        bbox_mins = vertices.min(axis=0).tolist()
        bbox_maxs = vertices.max(axis=0).tolist()
    
    return bbox_mins, bbox_maxs
