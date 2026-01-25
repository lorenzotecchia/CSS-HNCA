"""Unit tests for Oja's rule."""

import numpy as np
import pytest

from src.learning.oja import apply_oja


class TestOja:
    """Tests for Oja's rule apply function."""

    def test_oja_decays_weights(self):
        """Oja's rule should decay weights based on activity."""
        weights = np.array([[0.0, 0.8], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_current = np.array([0, 1])  # Post neuron fires
        oja_alpha = 0.1

        result = apply_oja(weights, link_matrix, firing_current, oja_alpha)

        expected = 0.8 * (1 - 0.1 * 1**2)  # 0.8 * 0.9 = 0.72
        assert result[0, 1] == pytest.approx(expected)

    def test_oja_no_change_when_disabled(self):
        """No change when oja_alpha is 0."""
        weights = np.array([[0.0, 0.8], [0.0, 0.0]])
        link_matrix = np.array([[False, True], [False, False]])
        firing_current = np.array([0, 1])

        result = apply_oja(weights, link_matrix, firing_current, 0.0)

        np.testing.assert_array_equal(result, weights)

    def test_oja_affects_all_incoming(self):
        """Oja affects all incoming weights to firing neurons."""
        weights = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        link_matrix = np.array([[False, True, True], [False, False, False], [False, False, False]])
        firing_current = np.array([0, 1, 0])  # Second neuron fires
        oja_alpha = 0.2

        result = apply_oja(weights, link_matrix, firing_current, oja_alpha)

        expected = 0.5 * (1 - 0.2 * 1**2)  # 0.5 * 0.8 = 0.4
        assert result[0, 1] == pytest.approx(expected)
        assert result[0, 2] == 0.5  # Not affected
        assert result[1, 1] == 0.0  # No link