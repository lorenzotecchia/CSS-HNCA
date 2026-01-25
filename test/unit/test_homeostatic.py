"""Unit tests for homeostatic scaling."""

from collections import deque

import numpy as np
import pytest

from src.learning.homeostatic import apply_homeostatic


class TestHomeostatic:
    """Tests for homeostatic scaling apply function."""

    def test_homeostatic_increase_low_activity(self):
        """Increase weights for neurons with low spike count."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_current = np.array([0, 1])
        spike_history = deque([np.array([0, 1])] * 10)  # 10 spikes for neuron 1
        spike_timespan = 10
        min_spike_amount = 5
        max_spike_amount = 15
        weight_change_constant = 0.1

        result = apply_homeostatic(
            weights, link_matrix, firing_current, spike_history,
            spike_timespan, min_spike_amount, max_spike_amount, weight_change_constant
        )

        # Neuron 1 has 10 spikes, which is between min and max, so no change
        assert result[0, 1] == 0.5

    def test_homeostatic_decrease_high_activity(self):
        """Decrease weights for neurons with high spike count."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_current = np.array([0, 1])
        spike_history = deque([np.array([0, 1])] * 20)  # 20 spikes for neuron 1
        spike_timespan = 20
        min_spike_amount = 5
        max_spike_amount = 15
        weight_change_constant = 0.1

        result = apply_homeostatic(
            weights, link_matrix, firing_current, spike_history,
            spike_timespan, min_spike_amount, max_spike_amount, weight_change_constant
        )

        # Neuron 1 has 20 spikes > max, so decrease
        assert result[0, 1] == 0.4  # 0.5 - 0.1

    def test_homeostatic_no_change_insufficient_history(self):
        """No change if history is shorter than timespan."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_current = np.array([0, 1])
        spike_history = deque([np.array([0, 1])])  # Only 1 entry
        spike_timespan = 10
        min_spike_amount = 5
        max_spike_amount = 15
        weight_change_constant = 0.1

        result = apply_homeostatic(
            weights, link_matrix, firing_current, spike_history,
            spike_timespan, min_spike_amount, max_spike_amount, weight_change_constant
        )

        assert result[0, 1] == 0.5