"""Unit tests for WeightUpdater."""

import numpy as np
import pytest

from src.learning.weight_update import WeightUpdater


class TestWeightUpdater:
    """Tests for WeightUpdater class."""

    def test_weight_updater_creation(self):
        """Should create WeightUpdater with default values."""
        updater = WeightUpdater()
        assert updater.enable_stdp is True
        assert updater.enable_oja is False
        assert updater.enable_homeostatic is False
        assert updater.learning_rate == 0.01

    def test_stdp_only(self):
        """Test with only STDP enabled."""
        updater = WeightUpdater(enable_stdp=True, enable_oja=False, enable_homeostatic=False)
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([1, 0])
        firing_current = np.array([0, 1])

        result = updater.apply(weights, link_matrix, firing_prev, firing_current)

        assert result[0, 1] > 0.5  # LTP increased

    def test_oja_only(self):
        """Test with only Oja enabled."""
        updater = WeightUpdater(enable_stdp=False, enable_oja=True, enable_homeostatic=False, oja_alpha=0.1)
        weights = np.array([[0.0, 0.8], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([0, 0])
        firing_current = np.array([0, 1])

        result = updater.apply(weights, link_matrix, firing_prev, firing_current)

        assert result[0, 1] < 0.8  # Oja decayed

    def test_homeostatic_only(self):
        """Test with only homeostatic enabled."""
        updater = WeightUpdater(
            enable_stdp=False, enable_oja=False, enable_homeostatic=True,
            spike_timespan=5, min_spike_amount=1, max_spike_amount=3, weight_change_constant=0.1
        )
        # Fill history with high activity
        for _ in range(5):
            updater.spike_history.append(np.array([0, 1]))

        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([0, 0])
        firing_current = np.array([0, 1])

        result = updater.apply(weights, link_matrix, firing_prev, firing_current)

        assert result[0, 1] < 0.5  # Homeostatic decreased

    def test_inhibitory_bounds(self):
        """Test that inhibitory weights are bounded correctly."""
        updater = WeightUpdater()
        weights = np.array([[0.0, -0.8], [0.0, 0.0]])  # Inhibitory weight
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([0, 0])
        firing_current = np.array([0, 0])

        result = updater.apply(weights, link_matrix, firing_prev, firing_current)

        assert -1.0 <= result[0, 1] <= -0.001

    def test_excitatory_bounds(self):
        """Test that excitatory weights are bounded correctly."""
        updater = WeightUpdater()
        weights = np.array([[0.0, 1.2], [0.0, 0.0]])  # Excitatory weight above max
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([0, 0])
        firing_current = np.array([0, 0])

        result = updater.apply(weights, link_matrix, firing_prev, firing_current)

        assert 0.0 <= result[0, 1] <= 1.0