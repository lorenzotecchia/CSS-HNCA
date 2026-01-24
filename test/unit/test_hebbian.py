"""Unit tests for Hebbian learning with STDP.

RED phase: These tests should fail until hebbian.py is implemented.

STDP Rules:
- LTP: w_AB += l if A fires at t and B fires at t+1
- LTD: w_AB -= f if B fires at t and A fires at t+1
- Weights bounded by [w_min, w_max]
"""

import numpy as np
import pytest

from src.learning.hebbian import HebbianLearner


class TestHebbianLearnerCreation:
    """Tests for HebbianLearner initialization."""

    def test_create_with_rates(self):
        """Should create learner with learning and forgetting rates."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
        )
        assert learner.learning_rate == 0.01
        assert learner.forgetting_rate == 0.005

    def test_create_with_bounds(self):
        """Should store weight bounds."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.1,
            weight_max=0.9,
        )
        assert learner.weight_min == 0.1
        assert learner.weight_max == 0.9


class TestLTPLongTermPotentiation:
    """Tests for LTP: weight increase when A(t)=1 and B(t+1)=1."""

    def test_ltp_increases_weight(self):
        """Weight should increase by learning_rate for causal firing."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        # A fired at t (prev), B fires at t+1 (current)
        # Connection A->B should be strengthened
        weights = np.array([
            [0.0, 0.5, 0.0],  # A->B is 0.5
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([True, False, False])   # A fired at t
        firing_current = np.array([False, True, False])  # B fires at t+1

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # A->B should increase by 0.1
        assert new_weights[0, 1] == pytest.approx(0.6)

    def test_ltp_only_where_link_exists(self):
        """LTP should only apply where structural link exists."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, False, False],  # No link A->B
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([True, False, False])
        firing_current = np.array([False, True, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # No links, no changes
        assert np.all(new_weights == 0)


class TestLTDLongTermDepression:
    """Tests for LTD: weight decrease when B(t)=1 and A(t+1)=1."""

    def test_ltd_decreases_weight(self):
        """Weight should decrease by forgetting_rate for anti-causal firing."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        # B fired at t (prev), A fires at t+1 (current)
        # Connection A->B should be weakened
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([False, True, False])   # B fired at t
        firing_current = np.array([True, False, False])  # A fires at t+1

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # A->B should decrease by 0.05
        assert new_weights[0, 1] == pytest.approx(0.45)

    def test_ltd_only_where_link_exists(self):
        """LTD should only apply where structural link exists."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([False, True, False])
        firing_current = np.array([True, False, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert np.all(new_weights == 0)


class TestWeightBounds:
    """Tests for weight bounds enforcement."""

    def test_ltp_respects_max_bound(self):
        """LTP should not exceed weight_max."""
        learner = HebbianLearner(
            learning_rate=0.5,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.9, 0.0],  # Close to max
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([True, False, False])
        firing_current = np.array([False, True, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Should be clamped to 1.0, not 1.4
        assert new_weights[0, 1] == 1.0

    def test_ltd_respects_min_bound(self):
        """LTD should not go below weight_min."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.5,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.1, 0.0],  # Close to min
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([False, True, False])
        firing_current = np.array([True, False, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Should be clamped to 0.0, not -0.4
        assert new_weights[0, 1] == 0.0

    def test_custom_bounds_respected(self):
        """Custom bounds should be respected."""
        learner = HebbianLearner(
            learning_rate=0.5,
            forgetting_rate=0.5,
            weight_min=0.2,
            weight_max=0.8,
        )
        weights = np.array([
            [0.0, 0.75, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ])
        # LTP on A->B (0.75 + 0.5 should clamp to 0.8)
        # LTD on B->C (0.25 - 0.5 should clamp to 0.2)
        firing_prev = np.array([True, False, True])
        firing_current = np.array([False, True, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert new_weights[0, 1] == 0.8  # Clamped to max
        assert new_weights[1, 2] == 0.2  # Clamped to min


class TestCombinedEffects:
    """Tests for combined LTP and LTD effects."""

    def test_both_ltp_and_ltd_in_same_step(self):
        """Both LTP and LTD can occur in the same time step."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        # A->B: LTP (A fired, B fires now)
        # B->A: LTD (B fires now, A fired before - wait, this doesn't match)
        # Let me set up correctly:
        # A fires at t, B fires at t+1: LTP on A->B
        # C fires at t, A fires at t+1: LTD on A->C (since C fired before, A fires now)
        weights = np.array([
            [0.0, 0.5, 0.5],  # A->B, A->C
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, True],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([True, False, True])    # A and C fired at t
        firing_current = np.array([True, True, False])  # A and B fire at t+1

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # A->B: LTP (A(t)=1, B(t+1)=1) -> 0.5 + 0.1 = 0.6
        # A->C: LTD (C(t)=1, A(t+1)=1) -> 0.5 - 0.05 = 0.45
        assert new_weights[0, 1] == pytest.approx(0.6)
        assert new_weights[0, 2] == pytest.approx(0.45)

    def test_no_change_when_no_firing(self):
        """Weights should not change when neurons don't fire."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([False, False, False])
        firing_current = np.array([False, False, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # No firing, no change
        assert np.array_equal(new_weights, weights)


class TestWeightMatrixPreservation:
    """Tests for weight matrix integrity."""

    def test_returns_new_array(self):
        """apply() should return a new array, not modify in place."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        original = weights.copy()
        link_matrix = np.ones((3, 3), dtype=bool)
        np.fill_diagonal(link_matrix, False)
        firing_prev = np.array([True, False, False])
        firing_current = np.array([False, True, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Original should be unchanged
        assert np.array_equal(weights, original)
        # New should be different
        assert not np.array_equal(new_weights, weights)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.zeros((10, 10))
        link_matrix = np.ones((10, 10), dtype=bool)
        np.fill_diagonal(link_matrix, False)
        firing_prev = np.random.choice([True, False], 10)
        firing_current = np.random.choice([True, False], 10)

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert new_weights.shape == weights.shape


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_neuron(self):
        """Should handle single neuron (no connections possible)."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([[0.0]])
        link_matrix = np.array([[False]])
        firing_prev = np.array([True])
        firing_current = np.array([True])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert new_weights[0, 0] == 0.0

    def test_zero_learning_rate(self):
        """With zero learning rate, no LTP should occur."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([True, False, False])
        firing_current = np.array([False, True, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # No LTP, weight stays same
        assert new_weights[0, 1] == 0.5

    def test_zero_forgetting_rate(self):
        """With zero forgetting rate, no LTD should occur."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
        )
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        link_matrix = np.array([
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ])
        firing_prev = np.array([False, True, False])
        firing_current = np.array([True, False, False])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # No LTD, weight stays same
        assert new_weights[0, 1] == 0.5


class TestPerTypeWeightBounds:
    """Tests for per-type (excitatory/inhibitory) weight bounds."""

    def test_learner_accepts_inhibitory_bounds(self):
        """HebbianLearner should accept weight_min_inh and weight_max_inh."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=0.3,
            weight_min_inh=-0.3,
            weight_max_inh=0.0,
        )
        assert learner.weight_min_inh == -0.3
        assert learner.weight_max_inh == 0.0

    def test_apply_enforces_excitatory_bounds(self):
        """Excitatory neurons should be clamped to [weight_min, weight_max]."""
        learner = HebbianLearner(
            learning_rate=0.5,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=0.2,
            weight_min_inh=-0.2,
            weight_max_inh=0.0,
        )
        weights = np.array([[0.0, 0.15], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        neuron_types = np.array([True, True])  # Both excitatory
        firing_prev = np.array([True, False])
        firing_curr = np.array([False, True])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr, neuron_types)
        assert new_weights[0, 1] <= 0.2  # Clamped to max

    def test_apply_enforces_inhibitory_bounds(self):
        """Inhibitory neurons should be clamped to [weight_min_inh, weight_max_inh]."""
        learner = HebbianLearner(
            learning_rate=0.5,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=0.2,
            weight_min_inh=-0.2,
            weight_max_inh=0.0,
        )
        weights = np.array([[0.0, -0.1], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        neuron_types = np.array([False, True])  # First inhibitory
        firing_prev = np.array([True, False])
        firing_curr = np.array([False, True])

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr, neuron_types)
        assert -0.2 <= new_weights[0, 1] <= 0.0

    def test_apply_backward_compatible_without_neuron_types(self):
        """apply() should work without neuron_types for backward compatibility."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.05,
            weight_min=0.0,
            weight_max=0.3,
        )
        weights = np.array([[0.0, 0.1], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([True, False])
        firing_curr = np.array([False, True])

        # Should not raise - uses excitatory bounds only
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr)
        assert new_weights is not None
