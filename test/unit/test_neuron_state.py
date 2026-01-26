"""Unit tests for NeuronState data structure.

RED phase: These tests should fail until NeuronState is implemented.
"""

import numpy as np
import pytest

from src.core.neuron_state import NeuronState


class TestNeuronStateCreation:
    """Tests for NeuronState.create factory method."""

    def test_create_returns_neuron_state(self):
        """create should return a NeuronState instance."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )
        assert isinstance(state, NeuronState)

    def test_threshold_stored_correctly(self):
        """NeuronState should store the threshold value."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.75,
            firing_count=1,
            seed=42,
        )
        assert state.threshold == 0.75


class TestNeuronStateFiringArrays:
    """Tests for firing state arrays."""

    def test_firing_shape(self):
        """Firing array should have shape (N,) for N neurons."""
        state = NeuronState.create(
            n_neurons=25,
            threshold=0.5,
            firing_count=3,
            seed=42,
        )
        assert state.firing.shape == (25,)

    def test_firing_dtype_bool(self):
        """Firing array should be boolean."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )
        assert state.firing.dtype == np.bool_

    def test_firing_prev_shape(self):
        """Previous firing array should have shape (N,)."""
        state = NeuronState.create(
            n_neurons=25,
            threshold=0.5,
            firing_count=3,
            seed=42,
        )
        assert state.firing_prev.shape == (25,)

    def test_firing_prev_dtype_bool(self):
        """Previous firing array should be boolean."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )
        assert state.firing_prev.dtype == np.bool_


class TestNeuronStateInitialFiring:
    """Tests for initial firing state with specified count."""

    def test_firing_count_zero(self):
        """With count=0, no neurons should fire initially."""
        state = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=0,
            seed=42,
        )
        assert not np.any(state.firing)

    def test_firing_count_all(self):
        """With count=n_neurons, all neurons should fire initially."""
        state = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=100,
            seed=42,
        )
        assert np.all(state.firing)

    def test_firing_count_exact(self):
        """With count=30, exactly 30 neurons should fire."""
        state = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=30,
            seed=42,
        )
        assert np.sum(state.firing) == 30

    def test_firing_prev_initially_false(self):
        """Previous firing state should start all False."""
        state = NeuronState.create(
            n_neurons=50,
            threshold=0.5,
            firing_count=25,
            seed=42,
        )
        assert not np.any(state.firing_prev)


class TestNeuronStateThresholdValidation:
    """Tests for threshold parameter validation."""

    def test_threshold_positive(self):
        """Threshold must be positive."""
        with pytest.raises(ValueError):
            NeuronState.create(
                n_neurons=10,
                threshold=-0.1,
                firing_count=1,
                seed=42,
            )

    def test_threshold_zero_allowed(self):
        """Threshold of zero should be allowed (edge case)."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.0,
            firing_count=1,
            seed=42,
        )
        assert state.threshold == 0.0


class TestNeuronStateFiringCountValidation:
    """Tests for firing_count validation."""

    def test_firing_count_negative_invalid(self):
        """Firing count must be >= 0."""
        with pytest.raises(ValueError):
            NeuronState.create(
                n_neurons=10,
                threshold=0.5,
                firing_count=-1,
                seed=42,
            )

    def test_firing_count_above_n_neurons_invalid(self):
        """Firing count must be <= n_neurons."""
        with pytest.raises(ValueError):
            NeuronState.create(
                n_neurons=10,
                threshold=0.5,
                firing_count=15,
                seed=42,
            )


class TestNeuronStateReproducibility:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_same_initial_firing(self):
        """Same seed should produce identical initial firing pattern."""
        state1 = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=30,
            seed=12345,
        )
        state2 = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=30,
            seed=12345,
        )
        assert np.array_equal(state1.firing, state2.firing)

    def test_different_seed_different_initial_firing(self):
        """Different seeds should produce different firing patterns."""
        state1 = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=50,
            seed=111,
        )
        state2 = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            firing_count=50,
            seed=222,
        )
        # With 50 firing, it's extremely unlikely to be identical
        assert not np.array_equal(state1.firing, state2.firing)


class TestNeuronStateUpdate:
    """Tests for update_firing method."""

    def test_update_firing_modifies_state(self):
        """update_firing should modify the firing state based on input."""
        state = NeuronState.create(
            n_neurons=5,
            threshold=0.5,
            firing_count=0,
            seed=42,
        )
        # Input signal exceeds threshold for neurons 0 and 2
        input_signal = np.array([0.6, 0.3, 0.8, 0.1, 0.4])
        state.update_firing(input_signal)
        expected = np.array([True, False, True, False, False])
        assert np.array_equal(state.firing, expected)

    def test_update_firing_preserves_previous(self):
        """update_firing should copy current to previous before update."""
        state = NeuronState.create(
            n_neurons=3,
            threshold=0.5,
            firing_count=0,
            seed=42,
        )
        # First update
        state.update_firing(np.array([0.6, 0.3, 0.8]))
        first_firing = state.firing.copy()

        # Second update
        state.update_firing(np.array([0.2, 0.7, 0.1]))

        # Previous should match first firing pattern
        assert np.array_equal(state.firing_prev, first_firing)

    def test_update_firing_threshold_boundary(self):
        """Exactly at threshold should fire (>= threshold)."""
        state = NeuronState.create(
            n_neurons=3,
            threshold=0.5,
            firing_count=0,
            seed=42,
        )
        input_signal = np.array([0.5, 0.499999, 0.500001])
        state.update_firing(input_signal)
        expected = np.array([True, False, True])
        assert np.array_equal(state.firing, expected)
