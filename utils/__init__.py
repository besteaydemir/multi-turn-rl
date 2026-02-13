# Utils package for 3D reasoning pipeline
from .rendering import (
    compute_visibility_score,
    compute_laplacian_variance_score,
    select_best_initial_view,
    render_mesh_from_pose,
)
from .instructions import (
    build_instruction_text,
    build_instruction_natural,
)
from .parsing import (
    extract_first_json,
    parse_qwen_output_and_get_movement,
    validate_rotation_matrix,
    validate_translation_vector,
)
from .camera import (
    look_at_camera_pose_center_from_forward,
    compute_initial_camera_pose,
    parse_rotation_angle,
    apply_movement_in_camera_frame,
    save_matrix,
)
from .data import (
    get_metadata_df,
    sky_direction_to_up_vector,
    get_sky_direction_for_scene,
    load_vsi_bench_questions,
    MCA_QUESTION_TYPES,
    NUMERICAL_QUESTION_TYPES,
    ALL_SEQUENTIAL_QUESTION_TYPES,
)
from .common import (
    timestamp_str,
)
from .mesh import (
    find_mesh_file,
    load_mesh_cached,
    clear_mesh_cache,
    get_mesh_bounds,
)
from .evaluation import (
    calculate_mra,
    evaluate_answer,
    compute_accuracy_by_type,
)
from .visualization import (
    render_birds_eye_view_with_path,
)
from .inference import (
    InferenceBackend,
    HuggingFaceBackend,
    VLLMBackend,
    create_inference_backend,
)

__all__ = [
    # rendering
    "compute_visibility_score",
    "compute_laplacian_variance_score",
    "select_best_initial_view",
    "render_mesh_from_pose",
    # instructions
    "build_instruction_text",
    "build_instruction_natural",
    # parsing
    "extract_first_json",
    "parse_qwen_output_and_get_movement",
    "validate_rotation_matrix",
    "validate_translation_vector",
    # camera
    "look_at_camera_pose_center_from_forward",
    "compute_initial_camera_pose",
    "parse_rotation_angle",
    "apply_movement_in_camera_frame",
    "save_matrix",
    # data
    "get_metadata_df",
    "sky_direction_to_up_vector",
    "get_sky_direction_for_scene",
    "load_vsi_bench_questions",
    "MCA_QUESTION_TYPES",
    "NUMERICAL_QUESTION_TYPES",
    "ALL_SEQUENTIAL_QUESTION_TYPES",
    # common
    "timestamp_str",
    # mesh
    "find_mesh_file",
    "load_mesh_cached",
    "clear_mesh_cache",
    "get_mesh_bounds",
    # evaluation
    "calculate_mra",
    "evaluate_answer",
    "compute_accuracy_by_type",
    # visualization
    "render_birds_eye_view_with_path",
    # inference backends
    "InferenceBackend",
    "HuggingFaceBackend",
    "VLLMBackend",
    "create_inference_backend",
]
