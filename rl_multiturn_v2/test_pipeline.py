#!/usr/bin/env python3
"""
Tests for rl_multiturn_v2 module.

Run with:
    python test_pipeline.py
    
Or with pytest:
    pytest test_pipeline.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TEST DATA STRUCTURES
# =============================================================================

def test_camera_pose():
    """Test CameraPose creation and serialization."""
    from rl_multiturn_v2.data_structures import CameraPose
    import numpy as np
    
    # Test with position/rotation
    pose1 = CameraPose(
        position=[1.0, 2.0, 3.0],
        rotation=[0.0, 0.0, 0.0, 1.0],
        fov=60.0,
    )
    
    assert np.allclose(pose1.position, [1.0, 2.0, 3.0])
    assert pose1.fov == 60.0
    
    # Test serialization
    d = pose1.to_dict()
    pose2 = CameraPose.from_dict(d)
    assert np.allclose(pose2.position, pose1.position)
    assert pose2.fov == pose1.fov
    
    print("✓ CameraPose tests passed")


def test_action():
    """Test Action creation."""
    from rl_multiturn_v2.data_structures import Action, CameraPose
    
    pose = CameraPose(position=[1.0, 2.0, 3.0], rotation=[0, 0, 0, 1], fov=60)
    action = Action(
        camera_pose=pose,
        raw_json='{"camera_pose": [[1,0,0,0]]}',
        parse_success=True,
    )
    
    assert action.parse_success
    assert action.camera_pose.position[0] == 1.0
    
    print("✓ Action tests passed")


def test_turn():
    """Test Turn creation."""
    from rl_multiturn_v2.data_structures import Turn, Action, CameraPose
    import torch
    
    pose = CameraPose(position=[0, 0, 0], rotation=[0, 0, 0, 1], fov=60)
    action = Action(camera_pose=pose, raw_json="{}", parse_success=True)
    
    turn = Turn(
        turn_index=0,
        generated_ids=torch.tensor([1, 2, 3, 4, 5]),
        logprobs=torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0]),
        action_token_mask=torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
        action=action,
    )
    
    assert turn.turn_index == 0
    assert turn.action_token_mask.sum() == 2
    
    print("✓ Turn tests passed")


def test_trajectory():
    """Test Trajectory creation and serialization."""
    from rl_multiturn_v2.data_structures import Trajectory, Turn, Action, CameraPose
    import torch
    
    # Create a trajectory with 2 turns
    turns = []
    for i in range(2):
        pose = CameraPose(position=[i, 0, 0], rotation=[0, 0, 0, 1], fov=60)
        action = Action(camera_pose=pose, raw_json="{}", parse_success=True)
        turn = Turn(
            turn_index=i,
            generated_ids=torch.tensor([1, 2, 3]),
            logprobs=torch.tensor([-1.0, -1.0, -1.0]),
            action_token_mask=torch.tensor([0, 1, 0], dtype=torch.bool),
            action=action,
            reward=0.0,
        )
        turns.append(turn)
    
    traj = Trajectory(
        trajectory_id="test_traj_001",
        question="What color is the chair?",
        turns=turns,
    )
    
    assert len(traj) == 2
    assert traj.trajectory_id == "test_traj_001"
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "traj.pt"
        traj.save(save_path)
        
        loaded = Trajectory.load(save_path)
        assert loaded.trajectory_id == traj.trajectory_id
        assert len(loaded) == len(traj)
    
    print("✓ Trajectory tests passed")


# =============================================================================
# TEST OUTPUT PARSER
# =============================================================================

def test_output_parser():
    """Test output parsing."""
    from rl_multiturn_v2.output_parser import OutputParser
    
    parser = OutputParser()
    
    # Test full output
    text = """[STATE]
I can see a room with a table and chairs.

[PLAN]
I need to look from another angle to see the door.

[PREDICT]
The next view should show the door location.

[ACTION]
{
    "camera_pose": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
    "fov": 60
}
"""
    
    result = parser.parse(text)
    
    assert "STATE" in result.sections
    assert "PLAN" in result.sections
    assert "ACTION" in result.sections
    assert result.action is not None
    assert result.action.parse_success
    
    print("✓ OutputParser tests passed")


def test_output_parser_final_answer():
    """Test parsing final answer on last turn."""
    from rl_multiturn_v2.output_parser import OutputParser
    
    parser = OutputParser()
    
    text = """[STATE]
Based on all my observations...

[FINAL_ANSWER]
A
"""
    
    result = parser.parse(text)
    
    assert result.final_answer is not None
    assert result.final_answer.answer_text == "A"
    
    print("✓ Final answer parsing tests passed")


def test_action_token_masker():
    """Test action token masking."""
    from rl_multiturn_v2.output_parser import ActionTokenMasker, ParseResult
    from rl_multiturn_v2.data_structures import Action, CameraPose
    import torch
    
    # Create mock tokenizer first
    class MockTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {
                0: "Hello ", 
                1: "[ACTION]\n", 
                2: "{", 
                3: '"camera_pose"', 
                4: ": ", 
                5: "[", 
                6: "]", 
                7: "}", 
                8: "\nEnd"
            }
            if len(ids) == 1:
                return mapping.get(ids[0].item() if torch.is_tensor(ids[0]) else ids[0], "")
            return "Hello [ACTION]\n{\"camera_pose\": []}\nEnd"
    
    tokenizer = MockTokenizer()
    masker = ActionTokenMasker(tokenizer)
    
    # Simulate tokenizer behavior
    text = "Hello [ACTION]\n{\"camera_pose\": []}\nEnd"
    token_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    # Create a ParseResult with action positions
    parse_result = ParseResult(
        sections={"ACTION": "{\"camera_pose\": []}"},
        action_json_start=17,  # Position of '{' 
        action_json_end=35,    # Position after '}'
        action=Action(camera_pose=CameraPose(position=[0,0,0]), raw_json="{}", parse_success=True),
        final_answer=None
    )
    
    # Note: This test is simplified; real masker needs proper character mapping
    mask, start_idx, end_idx = masker.create_mask(text, token_ids, parse_result)
    
    assert mask.shape == token_ids.shape
    
    print("✓ ActionTokenMasker tests passed")


def test_create_prompts():
    """Test prompt creation functions."""
    from rl_multiturn_v2.output_parser import create_system_prompt, create_turn_prompt
    
    system_prompt = create_system_prompt()
    assert "visual spatial reasoning" in system_prompt.lower()
    assert "camera" in system_prompt.lower()
    
    turn_prompt = create_turn_prompt(
        question="What color is the chair?",
        image_paths=["image1.png"],
        turn_number=1,
        max_turns=5,
        is_final_turn=False,
    )
    
    assert "What color is the chair?" in turn_prompt
    assert "turn 1" in turn_prompt.lower() or "Turn 1" in turn_prompt
    assert "[STATE]" in turn_prompt  # This should be in turn prompts
    
    print("✓ Prompt creation tests passed")


# =============================================================================
# TEST ROLLOUT ENGINE
# =============================================================================

def test_mock_rollout_engine():
    """Test mock rollout engine."""
    from rl_multiturn_v2.rollout import RolloutConfig, MockRolloutEngine
    
    config = RolloutConfig(max_turns=3)
    engine = MockRolloutEngine(config)
    
    trajectory = engine.collect_trajectory(
        question="What is in the room?",
        choices=["A chair", "A table", "A lamp", "A sofa"],
        scene_id="test_scene",
    )
    
    assert trajectory.question == "What is in the room?"
    assert len(trajectory.turns) == 3
    
    # Check that each turn has proper structure
    for i, turn in enumerate(trajectory.turns):
        assert turn.turn_index == i
        assert turn.generated_ids is not None
        assert turn.action_token_mask is not None
    
    print("✓ MockRolloutEngine tests passed")


def test_rollout_batch():
    """Test batch trajectory collection."""
    from rl_multiturn_v2.rollout import RolloutConfig, MockRolloutEngine
    
    config = RolloutConfig(max_turns=2)
    engine = MockRolloutEngine(config)
    
    questions = [
        {"question": "Q1", "choices": ["A"], "scene_id": "s1"},
        {"question": "Q2", "choices": ["B"], "scene_id": "s2"},
    ]
    
    # Collect trajectories individually
    trajectories = []
    for q in questions:
        traj = engine.collect_trajectory(
            question=q["question"],
            choices=q["choices"],
            scene_id=q["scene_id"],
        )
        trajectories.append(traj)
    
    assert len(trajectories) == 2
    assert trajectories[0].question == "Q1"
    assert trajectories[1].question == "Q2"
    
    print("✓ Batch rollout tests passed")


# =============================================================================
# TEST TRAINER
# =============================================================================

def test_compute_advantages():
    """Test advantage computation."""
    from rl_multiturn_v2.trainer import compute_advantages
    from rl_multiturn_v2.data_structures import Trajectory, Turn, Action, CameraPose
    import torch
    
    # Create trajectory with rewards
    turns = []
    for i in range(3):
        pose = CameraPose(position=[0, 0, 0], rotation=[0, 0, 0, 1], fov=60)
        action = Action(camera_pose=pose, raw_json="{}", parse_success=True)
        turn = Turn(
            turn_index=i,
            generated_ids=torch.tensor([1, 2, 3]),
            logprobs=torch.tensor([-1.0, -1.0, -1.0]),
            action_token_mask=torch.tensor([1, 1, 1], dtype=torch.bool),
            action=action,
            reward=0.0,  # Zero rewards as per spec
        )
        turns.append(turn)
    
    traj = Trajectory(trajectory_id="test", question="Q", turns=turns)
    
    advantages = compute_advantages([traj])
    
    # Should return a list of lists (one per trajectory)
    assert len(advantages) == 1  # One trajectory
    assert len(advantages[0]) == 3  # Three turns
    
    print("✓ Advantage computation tests passed")


def test_trainer_config():
    """Test trainer configuration."""
    from rl_multiturn_v2.trainer import TrainerConfig
    
    config = TrainerConfig(
        learning_rate=1e-5,
        use_ppo=True,
        ppo_clip_range=0.2,
    )
    
    assert config.learning_rate == 1e-5
    assert config.use_ppo
    assert config.ppo_clip_range == 0.2
    
    print("✓ TrainerConfig tests passed")


def test_rl_trainer():
    """Test RLTrainer with mock model."""
    import torch
    from rl_multiturn_v2.trainer import TrainerConfig, RLTrainer
    
    # Create a simple mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(10, 100)
        
        def forward(self, input_ids=None, **kwargs):
            return type('Output', (), {'logits': torch.randn(1, 5, 100)})()
    
    model = MockModel()
    config = TrainerConfig(learning_rate=1e-5, use_wandb=False, device="cpu")  # Use CPU for testing
    
    trainer = RLTrainer(model=model, config=config)
    
    assert trainer.model is model
    assert trainer.optimizer is not None
    
    print("✓ RLTrainer tests passed")


# =============================================================================
# TEST LOGGING
# =============================================================================

def test_logging_dataclasses():
    """Test logging data classes."""
    from rl_multiturn_v2.logging_utils import TurnLog, TrajectoryLog, UpdateLog
    
    turn = TurnLog(
        turn_index=0,
        num_tokens=50,
        num_action_tokens=10,
        num_reasoning_tokens=40,
        action_json='{"camera_pose": []}',
    )
    
    traj = TrajectoryLog(
        trajectory_id="traj_001",
        scene_id="test_scene",
        num_views=5,
        total_tokens=250,
        total_action_tokens=50,
        total_reasoning_tokens=200,
        final_answer="A",
        is_correct=True,
    )
    
    update = UpdateLog(
        update_step=1,
        policy_loss=0.5,
        num_trajectories=4,
    )
    
    assert turn.turn_index == 0
    assert traj.num_views == 5
    assert traj.total_tokens == 250
    assert update.policy_loss == 0.5
    
    print("✓ Logging dataclasses tests passed")


def test_file_logger():
    """Test file logger."""
    from rl_multiturn_v2.logging_utils import FileLogger, UpdateLog
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = FileLogger(tmpdir)
        
        update = UpdateLog(update_step=1, policy_loss=0.5, num_trajectories=4)
        logger.log_update(update)
        
        # Check file was created
        log_file = Path(tmpdir) / "update_logs.jsonl"  # Correct filename
        assert log_file.exists()
        
        # Check content
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["update_step"] == 1
    
    print("✓ FileLogger tests passed")


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_end_to_end_mock():
    """End-to-end test with mock components."""
    from rl_multiturn_v2.rollout import RolloutConfig, MockRolloutEngine
    from rl_multiturn_v2.trainer import TrainerConfig, compute_advantages
    from rl_multiturn_v2.logging_utils import Logger, log_trajectory
    import torch
    import tempfile
    
    # Setup
    config = RolloutConfig(max_turns=3)
    engine = MockRolloutEngine(config)
    
    questions = [
        {"question": "Q1", "choices": ["A"], "scene_id": "s1"},
        {"question": "Q2", "choices": ["B"], "scene_id": "s2"},
    ]
    
    # Collect trajectories individually
    trajectories = []
    for q in questions:
        traj = engine.collect_trajectory(
            question=q["question"],
            choices=q["choices"],
            scene_id=q["scene_id"],
        )
        trajectories.append(traj)
    
    assert len(trajectories) == 2
    
    # Compute advantages
    advantages = compute_advantages(trajectories)
    assert len(advantages) == 2  # 2 trajectories
    assert len(advantages[0]) == 3  # 3 turns each
    
    # Log
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, use_wandb=False)
        
        # Log all trajectories at once
        logger.log_trajectories(trajectories, step=0)
        
        logger.finish()
    
    print("✓ End-to-end mock test passed")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running rl_multiturn_v2 Tests")
    print("=" * 60)
    
    # Data structures
    test_camera_pose()
    test_action()
    test_turn()
    test_trajectory()
    
    # Output parser
    test_output_parser()
    test_output_parser_final_answer()
    test_action_token_masker()
    test_create_prompts()
    
    # Rollout
    test_mock_rollout_engine()
    test_rollout_batch()
    
    # Trainer
    test_compute_advantages()
    test_trainer_config()
    test_rl_trainer()
    
    # Logging
    test_logging_dataclasses()
    test_file_logger()
    
    # Integration
    test_end_to_end_mock()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
