#!/usr/bin/env python3
"""
Tests for trajectory dataclasses.

Run: pytest vagen_vsi_rl/tests/test_trajectory.py -v
"""

import pytest
import json
import tempfile
from pathlib import Path

import torch
from vagen_vsi_rl.rollout.trajectory import Turn, Trajectory


class TestTurn:
    """Tests for Turn dataclass."""

    def test_default_values(self):
        """Turn should have sensible defaults."""
        turn = Turn(turn_index=0)
        
        assert turn.turn_index == 0
        assert turn.image_paths == []
        assert turn.reward == 0.0
        assert turn.done_flag is False

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        turn = Turn(
            turn_index=1,
            generated_text='{"answer": "A"}',
            reward=0.5,
            done_flag=True,
        )
        
        d = turn.to_dict()
        assert d["turn_index"] == 1
        assert d["reward"] == 0.5
        
        # Should be JSON-serializable
        json.dumps(d)

    def test_from_dict(self):
        """from_dict should reconstruct Turn."""
        original = Turn(
            turn_index=2,
            generated_text="test",
            reward=1.0,
        )
        d = original.to_dict()
        restored = Turn.from_dict(d)
        
        assert restored.turn_index == 2
        assert restored.generated_text == "test"
        assert restored.reward == 1.0


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_len(self):
        """len() should return number of turns."""
        traj = Trajectory(
            turns=[Turn(0), Turn(1), Turn(2)]
        )
        assert len(traj) == 3

    def test_compute_returns_gamma_1(self):
        """compute_returns with gamma=1 should sum rewards."""
        traj = Trajectory(
            terminal_reward=0.0,
            turns=[
                Turn(0, reward=0.0),
                Turn(1, reward=0.0),
                Turn(2, reward=1.0),
            ]
        )
        returns = traj.compute_returns(gamma=1.0)
        
        # From end: [1, 1, 1] (all get full terminal reward)
        assert returns == [1.0, 1.0, 1.0]

    def test_compute_returns_discounted(self):
        """compute_returns with gamma<1 should discount."""
        traj = Trajectory(
            terminal_reward=0.0,
            turns=[
                Turn(0, reward=0.0),
                Turn(1, reward=0.0),
                Turn(2, reward=1.0),
            ]
        )
        returns = traj.compute_returns(gamma=0.9)
        
        # G_2 = 1.0
        # G_1 = 0.0 + 0.9 * 1.0 = 0.9
        # G_0 = 0.0 + 0.9 * 0.9 = 0.81
        assert returns[2] == pytest.approx(1.0)
        assert returns[1] == pytest.approx(0.9)
        assert returns[0] == pytest.approx(0.81)

    def test_fill_returns(self):
        """fill_returns should write to Turn.returns."""
        traj = Trajectory(
            turns=[Turn(0, reward=0.5), Turn(1, reward=0.5)]
        )
        traj.fill_returns(gamma=1.0)
        
        assert traj.turns[0].returns is not None
        assert traj.turns[1].returns is not None

    def test_to_dict(self):
        """to_dict should serialize entire trajectory."""
        traj = Trajectory(
            trajectory_id="test_001",
            question="What?",
            ground_truth="A",
            final_answer="A",
            is_correct=True,
            turns=[Turn(0), Turn(1)],
        )
        
        d = traj.to_dict()
        assert d["trajectory_id"] == "test_001"
        assert d["is_correct"] is True
        assert len(d["turns"]) == 2
        
        # JSON-serializable
        json.dumps(d)

    def test_save_and_load(self):
        """save() and load() should work correctly."""
        traj = Trajectory(
            trajectory_id="test_save",
            question="Test question?",
            ground_truth="B",
            final_answer="B",
            is_correct=True,
            turns=[
                Turn(0, generated_text="move", reward=0.1),
                Turn(1, generated_text="answer", reward=1.0, done_flag=True),
            ],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "traj"
            traj.save(save_path)
            
            # Check file exists
            json_path = save_path / "trajectory.json"
            assert json_path.exists()
            
            # Load and verify
            loaded = Trajectory.load(save_path)
            assert loaded.trajectory_id == "test_save"
            assert loaded.is_correct is True
            assert len(loaded.turns) == 2

    def test_empty_trajectory(self):
        """Empty trajectory should be valid."""
        traj = Trajectory()
        
        assert len(traj) == 0
        assert traj.compute_returns() == []
