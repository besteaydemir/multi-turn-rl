#!/usr/bin/env python3
"""
Tests for advantage estimation (Step 7B, 9 deliverables).

Run: pytest vagen_vsi_rl/tests/test_advantage.py -v
"""

import pytest
from vagen_vsi_rl.rl.advantage import (
    compute_monte_carlo_returns,
    compute_gae,
    compute_bilevel_gae,
    compute_advantage,
)
from vagen_vsi_rl.rollout.trajectory import Turn, Trajectory


def make_trajectory(rewards: list) -> Trajectory:
    """Helper to create trajectory with given per-turn rewards."""
    traj = Trajectory(
        turns=[
            Turn(turn_index=i, reward=r) 
            for i, r in enumerate(rewards)
        ]
    )
    return traj


class TestMonteCarloReturns:
    """Tests for compute_monte_carlo_returns()."""

    def test_single_turn_returns(self):
        """Single turn: return = reward."""
        traj = make_trajectory([1.0])
        compute_monte_carlo_returns(traj, gamma=1.0, normalise=False)
        assert traj.turns[0].returns == 1.0

    def test_multi_turn_gamma_1(self):
        """With gamma=1, returns are cumulative sums from end."""
        traj = make_trajectory([0.0, 0.0, 1.0])
        compute_monte_carlo_returns(traj, gamma=1.0, normalise=False)
        assert traj.turns[0].returns == 1.0  # 0 + 0 + 1
        assert traj.turns[1].returns == 1.0  # 0 + 1
        assert traj.turns[2].returns == 1.0  # 1

    def test_discounting(self):
        """With gamma < 1, future rewards are discounted."""
        traj = make_trajectory([0.0, 0.0, 1.0])
        compute_monte_carlo_returns(traj, gamma=0.5, normalise=False)
        # G_2 = 1.0
        # G_1 = 0 + 0.5 * 1.0 = 0.5
        # G_0 = 0 + 0.5 * 0.5 = 0.25
        assert traj.turns[2].returns == pytest.approx(1.0)
        assert traj.turns[1].returns == pytest.approx(0.5)
        assert traj.turns[0].returns == pytest.approx(0.25)

    def test_normalisation(self):
        """Normalised advantages should have mean ≈ 0."""
        traj = make_trajectory([0.0, 0.0, 1.0])
        compute_monte_carlo_returns(traj, normalise=True)
        advantages = [t.advantage for t in traj.turns]
        assert pytest.approx(sum(advantages) / len(advantages), abs=1e-6) == 0.0


class TestGAE:
    """Tests for compute_gae()."""

    def test_gae_with_zero_values(self):
        """GAE with zero values should behave like MC returns."""
        traj = make_trajectory([0.0, 0.0, 1.0])
        values = [0.0, 0.0, 0.0]
        compute_gae(traj, values, gamma=1.0, lam=1.0, normalise=False)
        
        # With V=0 and lambda=1, GAE = MC returns
        assert traj.turns[0].advantage == pytest.approx(1.0)
        assert traj.turns[1].advantage == pytest.approx(1.0)
        assert traj.turns[2].advantage == pytest.approx(1.0)

    def test_gae_fills_values(self):
        """Values should be stored in turns."""
        traj = make_trajectory([0.0, 1.0])
        values = [0.5, 0.8]
        compute_gae(traj, values, normalise=False)
        
        assert traj.turns[0].value == 0.5
        assert traj.turns[1].value == 0.8

    def test_gae_computes_returns(self):
        """Returns = advantage + value."""
        traj = make_trajectory([0.0, 1.0])
        values = [0.5, 0.8]
        compute_gae(traj, values, normalise=False)
        
        for t in traj.turns:
            assert t.returns == pytest.approx(t.advantage + t.value)

    def test_wrong_values_length_raises(self):
        """Mismatched values length should raise."""
        traj = make_trajectory([0.0, 1.0])
        values = [0.5]  # Wrong length
        with pytest.raises(AssertionError):
            compute_gae(traj, values)


class TestBilevelGAE:
    """Tests for compute_bilevel_gae() (Step 9)."""

    def test_bilevel_gae_basic(self):
        """Bi-level GAE should fill advantages."""
        traj = make_trajectory([0.0, 0.0, 1.0])
        values = [0.0, 0.0, 0.0]
        compute_bilevel_gae(traj, values, normalise=False)
        
        # Should have advantages filled
        for t in traj.turns:
            assert t.advantage is not None

    def test_bilevel_gae_empty_trajectory(self):
        """Empty trajectory should be handled gracefully."""
        traj = Trajectory(turns=[])
        result = compute_bilevel_gae(traj, [], normalise=False)
        assert result is traj


class TestComputeAdvantage:
    """Tests for compute_advantage() unified interface."""

    def test_mc_method(self):
        """method='mc' should use Monte-Carlo."""
        traj = make_trajectory([0.0, 1.0])
        compute_advantage(traj, [0.0, 0.0], method="mc")
        assert all(t.returns is not None for t in traj.turns)

    def test_token_gae_method(self):
        """method='token_gae' should use standard GAE."""
        traj = make_trajectory([0.0, 1.0])
        compute_advantage(traj, [0.0, 0.0], method="token_gae")
        assert all(t.advantage is not None for t in traj.turns)

    def test_bilevel_gae_method(self):
        """method='bilevel_gae' should use bi-level GAE."""
        traj = make_trajectory([0.0, 1.0])
        compute_advantage(traj, [0.0, 0.0], method="bilevel_gae")
        assert all(t.advantage is not None for t in traj.turns)

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        traj = make_trajectory([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown advantage method"):
            compute_advantage(traj, [0.0, 0.0], method="invalid")
