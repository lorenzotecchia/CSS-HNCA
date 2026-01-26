"""Unit tests for STDP learning rule."""

import numpy as np
import pytest

from src.learning.stdp import apply_stdp


class TestSTDP:
    """Tests for STDP apply function."""

    def test_stdp_ltp_increases_weight(self):
        """LTP should increase weights when pre fires before post."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([1, 0])  # Pre fires
        firing_current = np.array([0, 1])  # Post fires

        result = apply_stdp(weights, link_matrix, firing_prev, firing_current, 0.1, 0.05)

        assert result[0, 1] == 0.6  # Increased by 0.1

    def test_stdp_ltd_decreases_weight(self):
        """LTD should decrease weights when post fires before pre."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_prev = np.array([0, 1])  # Post fires
        firing_current = np.array([1, 0])  # Pre fires

        result = apply_stdp(weights, link_matrix, firing_prev, firing_current, 0.1, 0.05)

        assert result[0, 1] == 0.45  # Decreased by 0.05

    def test_stdp_no_change_without_links(self):
        """No change for non-linked connections."""
        weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        link_matrix = np.array([[False, False], [False, False]])
        firing_prev = np.array([1, 1])
        firing_current = np.array([1, 1])

        result = apply_stdp(weights, link_matrix, firing_prev, firing_current, 0.1, 0.05)

        np.testing.assert_array_equal(result, weights)

    def test_stdp_no_self_connections(self):
        """Diagonal should remain unchanged when no self-links exist."""
        weights = np.eye(2) * 0.5
        link_matrix = np.ones((2, 2), dtype=bool)
        np.fill_diagonal(link_matrix, False)  # No self-connections
        firing_prev = np.array([1, 1])
        firing_current = np.array([1, 1])

        result = apply_stdp(weights, link_matrix, firing_prev, firing_current, 0.1, 0.05)

        assert result[0, 0] == 0.5
        assert result[1, 1] == 0.5