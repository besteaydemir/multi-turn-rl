#!/usr/bin/env python3
"""
Tests for reward functions (Step 3 deliverable).

Run: pytest vagen_vsi_rl/tests/test_rewards.py -v
"""

import pytest
from vagen_vsi_rl.rl.rewards import (
    compute_turn_reward,
    compute_terminal_reward,
    compute_rewards,
    RewardConfig,
    is_valid_action_json,
)
from vagen_vsi_rl.rollout.trajectory import Turn, Trajectory


class TestComputeTurnReward:
    """Tests for compute_turn_reward()."""

    def test_valid_action_gets_format_reward(self):
        """Valid parsed action should get format_reward bonus."""
        cfg = RewardConfig(format_reward=0.1, step_penalty=0.0)
        turn = Turn(
            turn_index=0,
            action_dict={"rotation_angle_degrees": 45, "forward_meters": 0.5},
        )
        r = compute_turn_reward(turn, cfg)
        assert r == 0.1

    def test_invalid_action_no_format_reward(self):
        """Unparseable action gets no format reward (but no penalty either)."""
        cfg = RewardConfig(format_reward=0.1, step_penalty=0.0)
        turn = Turn(turn_index=0, action_dict=None, done_flag=False)
        r = compute_turn_reward(turn, cfg)
        assert r == 0.0

    def test_done_action_gets_format_reward(self):
        """done=True action should get format reward."""
        cfg = RewardConfig(format_reward=0.1, step_penalty=0.0)
        turn = Turn(turn_index=0, action_dict=None, done_flag=True)
        r = compute_turn_reward(turn, cfg)
        assert r == 0.1

    def test_step_penalty_applied(self):
        """Step penalty should be added."""
        cfg = RewardConfig(format_reward=0.1, step_penalty=-0.02)
        turn = Turn(
            turn_index=0,
            action_dict={"rotation_angle_degrees": 0},
        )
        r = compute_turn_reward(turn, cfg)
        assert r == pytest.approx(0.08)  # 0.1 - 0.02


class TestComputeTerminalReward:
    """Tests for compute_terminal_reward()."""

    def test_correct_mcq_answer(self):
        """Correct MCQ answer gets positive reward."""
        cfg = RewardConfig(correct_answer=1.0, wrong_answer=-0.5)
        r = compute_terminal_reward("C", "C", is_numerical=False, config=cfg)
        assert r == 1.0

    def test_wrong_mcq_answer(self):
        """Wrong MCQ answer gets negative reward."""
        cfg = RewardConfig(correct_answer=1.0, wrong_answer=-0.5)
        r = compute_terminal_reward("A", "C", is_numerical=False, config=cfg)
        assert r == -0.5

    def test_case_insensitive_mcq(self):
        """MCQ comparison should be case-insensitive."""
        r = compute_terminal_reward("c", "C", is_numerical=False)
        assert r > 0  # Should be correct

    def test_no_answer_penalty(self):
        """No answer should get no_answer penalty."""
        cfg = RewardConfig(no_answer=-1.0)
        r = compute_terminal_reward(None, "C", is_numerical=False, config=cfg)
        assert r == -1.0

    def test_numerical_exact_match(self):
        """Exact numerical match should give full reward."""
        cfg = RewardConfig(correct_answer=1.0, wrong_answer=0.0)
        r = compute_terminal_reward("5.0", "5.0", is_numerical=True, config=cfg)
        assert r == pytest.approx(1.0)

    def test_numerical_close_match(self):
        """Close numerical match should give partial reward via MRA."""
        cfg = RewardConfig(correct_answer=1.0, wrong_answer=0.0)
        r = compute_terminal_reward("5.5", "5.0", is_numerical=True, config=cfg)
        # Should be somewhere between 0 and 1
        assert 0.0 < r < 1.0

    def test_numerical_far_off(self):
        """Way off numerical answer should give low reward."""
        cfg = RewardConfig(correct_answer=1.0, wrong_answer=0.0)
        r = compute_terminal_reward("100", "5.0", is_numerical=True, config=cfg)
        assert r < 0.5

    def test_reward_scale(self):
        """reward_scale should multiply the result."""
        cfg = RewardConfig(correct_answer=1.0, reward_scale=10.0)
        r = compute_terminal_reward("C", "C", is_numerical=False, config=cfg)
        assert r == 10.0


class TestComputeRewards:
    """Tests for compute_rewards() on full trajectory."""

    def test_fills_all_turn_rewards(self):
        """All turns should have reward filled."""
        traj = Trajectory(
            ground_truth="C",
            final_answer="C",
            turns=[
                Turn(turn_index=0, action_dict={"forward": 0.5}),
                Turn(turn_index=1, action_dict={"forward": 0.3}),
                Turn(turn_index=2, done_flag=True),
            ],
        )
        compute_rewards(traj)
        assert all(t.reward is not None for t in traj.turns)

    def test_terminal_reward_on_last_turn_only(self):
        """Terminal reward should only be added to last turn."""
        cfg = RewardConfig(correct_answer=1.0, format_reward=0.0, step_penalty=0.0)
        traj = Trajectory(
            ground_truth="C",
            final_answer="C",
            turns=[
                Turn(turn_index=0, action_dict={}),
                Turn(turn_index=1, done_flag=True),
            ],
        )
        compute_rewards(traj, cfg)
        assert traj.turns[0].reward == 0.0
        assert traj.turns[1].reward == 1.0


class TestIsValidActionJson:
    """Tests for is_valid_action_json()."""

    def test_valid_movement_json(self):
        text = '{"rotation_angle_degrees": 45, "forward_meters": 0.5}'
        assert is_valid_action_json(text) is True

    def test_valid_answer_json(self):
        text = '{"answer": "C", "done": true}'
        assert is_valid_action_json(text) is True

    def test_invalid_json(self):
        text = "not json at all"
        assert is_valid_action_json(text) is False

    def test_json_in_markdown(self):
        text = '```json\n{"forward_meters": 0.5}\n```'
        assert is_valid_action_json(text) is True

    def test_empty_json(self):
        text = "{}"
        assert is_valid_action_json(text) is False
