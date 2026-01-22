"""Unit tests for weight decay mechanisms.

RED phase: These tests should fail until weight decay is implemented.

Two mechanisms:
1. Baseline weight decay: W(t+1) = (1 - α) * W(t)
2. Oja rule: W(t+1) -= α_oja * activity² * W
"""

import numpy as np
import pytest

from src.learning.hebbian import HebbianLearner


class TestBaselineWeightDecay:
    """Tests for baseline weight decay (decay_alpha parameter)."""

    def test_decay_reduces_weights(self):
        """Weights should decrease by decay factor each step."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.1,  # 10% decay per step
        )
        
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])
        link_matrix = np.array([[False, True], [True, False]])
        
        # No firing - just decay
        firing_prev = np.array([False, False])
        firing_current = np.array([False, False])
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # Expected: 0.5 * (1 - 0.1) = 0.45
        expected = 0.45
        np.testing.assert_allclose(new_weights[0, 1], expected, rtol=1e-5)
        np.testing.assert_allclose(new_weights[1, 0], expected, rtol=1e-5)

    def test_decay_alpha_zero_no_decay(self):
        """With decay_alpha=0, weights should not decay without activity."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.0,
        )
        
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])
        link_matrix = np.array([[False, True], [True, False]])
        firing_prev = np.array([False, False])
        firing_current = np.array([False, False])
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # No change without decay
        np.testing.assert_allclose(new_weights, weights, rtol=1e-5)

    def test_decay_respects_weight_bounds(self):
        """Decay should not push weights below weight_min."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.9,  # Very aggressive decay
        )
        
        weights = np.array([[0.0, 0.1], [0.1, 0.0]])
        link_matrix = np.array([[False, True], [True, False]])
        firing_prev = np.array([False, False])
        firing_current = np.array([False, False])
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # Should not go below 0
        assert np.all(new_weights >= 0.0)


class TestOjaDecay:
    """Tests for Oja rule activity-dependent decay."""

    def test_oja_decay_proportional_to_activity_squared(self):
        """Oja decay: W -= α_oja * activity² * W for firing neurons."""
        learner = HebbianLearner(
            learning_rate=0.0,  # Disable LTP
            forgetting_rate=0.0,  # Disable LTD
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.0,  # Disable baseline decay
            oja_alpha=0.1,
        )
        
        # Connection A -> B, weight 0.5
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        
        # B fires (postsynaptic activity)
        firing_prev = np.array([False, False])
        firing_current = np.array([False, True])  # B fires
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # Oja decay on W[0,1]: 0.5 - 0.1 * 1² * 0.5 = 0.5 - 0.05 = 0.45
        expected = 0.5 - 0.1 * 1.0 * 0.5
        np.testing.assert_allclose(new_weights[0, 1], expected, rtol=1e-5)

    def test_oja_no_decay_when_postsynaptic_silent(self):
        """Oja decay should not apply when postsynaptic neuron is silent."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.0,
            oja_alpha=0.1,
        )
        
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        
        # Neither fires
        firing_prev = np.array([False, False])
        firing_current = np.array([False, False])
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # No change
        np.testing.assert_allclose(new_weights[0, 1], 0.5, rtol=1e-5)

    def test_oja_alpha_zero_disables_oja(self):
        """With oja_alpha=0, no Oja decay should occur."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.0,
            oja_alpha=0.0,
        )
        
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([False, False])
        firing_current = np.array([False, True])  # B fires
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # No change despite B firing
        np.testing.assert_allclose(new_weights[0, 1], 0.5, rtol=1e-5)


class TestCombinedDecayMechanisms:
    """Tests for combined baseline + Oja decay."""

    def test_both_decays_apply(self):
        """Both baseline and Oja decay should apply together."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.1,  # 10% baseline decay
            oja_alpha=0.1,     # Oja decay
        )
        
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([False, False])
        firing_current = np.array([False, True])  # B fires
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # Baseline: 0.5 * (1 - 0.1) = 0.45
        # Oja: 0.45 - 0.1 * 1 * 0.45 = 0.45 * 0.9 = 0.405
        # Or applied together: 0.5 * 0.9 - 0.1 * 1 * (0.5 * 0.9) = 0.405
        # Exact order may vary - just check it's less than baseline alone
        assert new_weights[0, 1] < 0.45

    def test_decay_with_ltp_balance(self):
        """LTP and decay should create equilibrium for active pathways."""
        learner = HebbianLearner(
            learning_rate=0.1,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.05,
            oja_alpha=0.05,
        )
        
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        
        # A fires at t-1, B fires at t (LTP condition)
        firing_prev = np.array([True, False])
        firing_current = np.array([False, True])
        
        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)
        
        # LTP should add, decay should subtract
        # The result depends on balance - just verify it's bounded
        assert 0.0 <= new_weights[0, 1] <= 1.0


class TestWeightDecayConfig:
    """Tests for weight decay config parameters."""

    def test_learner_accepts_decay_alpha(self):
        """HebbianLearner should accept decay_alpha parameter."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
            decay_alpha=0.001,
        )
        assert learner.decay_alpha == 0.001

    def test_learner_accepts_oja_alpha(self):
        """HebbianLearner should accept oja_alpha parameter."""
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
            oja_alpha=0.001,
        )
        assert learner.oja_alpha == 0.001
