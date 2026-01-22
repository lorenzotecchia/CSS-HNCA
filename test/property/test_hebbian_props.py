"""Property-based tests for Hebbian learning using Hypothesis.

Tests invariants that must hold for any valid input combinations.
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.learning.hebbian import HebbianLearner


# Custom strategies
n_neurons_strategy = st.integers(min_value=2, max_value=50)
rate_strategy = st.floats(min_value=0.0, max_value=0.5, allow_nan=False)
bound_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


def create_test_inputs(n_neurons: int, seed: int):
    """Create random test inputs for learner."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0, 1, (n_neurons, n_neurons))
    np.fill_diagonal(weights, 0)  # No self-connections
    link_matrix = rng.random((n_neurons, n_neurons)) > 0.5
    np.fill_diagonal(link_matrix, False)
    # Zero weights where no links
    weights = np.where(link_matrix, weights, 0.0)
    firing_prev = rng.random(n_neurons) > 0.5
    firing_current = rng.random(n_neurons) > 0.5
    return weights, link_matrix, firing_prev, firing_current


class TestWeightBoundsInvariant:
    """Property tests for weight bounds."""

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        weight_min=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        weight_max=st.floats(min_value=0.7, max_value=1.0, allow_nan=False),
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_weights_always_within_bounds(
        self, n_neurons, learning_rate, forgetting_rate, weight_min, weight_max, seed
    ):
        """Weights where links exist must be within [weight_min, weight_max]."""
        assume(weight_min < weight_max)

        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=weight_min,
            weight_max=weight_max,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )
        # Scale weights to be within bounds where links exist
        weights = np.where(
            link_matrix,
            weights * (weight_max - weight_min) + weight_min,
            0.0
        )

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Only check bounds where links exist (non-linked should be 0)
        linked_weights = new_weights[link_matrix]
        if len(linked_weights) > 0:
            assert np.all(linked_weights >= weight_min)
            assert np.all(linked_weights <= weight_max)
        # Non-linked should be 0
        assert np.all(new_weights[~link_matrix] == 0)

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
        n_iterations=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30)
    def test_bounds_hold_over_iterations(
        self, n_neurons, learning_rate, forgetting_rate, seed, n_iterations
    ):
        """Bounds must hold after multiple learning applications."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, _, _ = create_test_inputs(n_neurons, seed)
        rng = np.random.default_rng(seed)

        for _ in range(n_iterations):
            firing_prev = rng.random(n_neurons) > 0.5
            firing_current = rng.random(n_neurons) > 0.5
            weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)


class TestStructuralConstraints:
    """Property tests for structural connectivity constraints."""

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_no_weights_where_no_links(
        self, n_neurons, learning_rate, forgetting_rate, seed
    ):
        """Weights must remain zero where no structural link exists."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Where there's no link, weight must be 0
        no_link_weights = new_weights[~link_matrix]
        assert np.all(no_link_weights == 0)

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_output_shape_preserved(
        self, n_neurons, learning_rate, forgetting_rate, seed
    ):
        """Output shape must match input shape."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert new_weights.shape == weights.shape


class TestLearningBehavior:
    """Property tests for learning rule behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_no_change_with_zero_rates(self, n_neurons, seed):
        """With zero learning and forgetting rates, weights should not change."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=0.0,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert np.allclose(new_weights, weights)

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=st.floats(min_value=0.01, max_value=0.1, allow_nan=False),
        forgetting_rate=st.floats(min_value=0.01, max_value=0.1, allow_nan=False),
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_no_change_when_no_firing(self, n_neurons, learning_rate, forgetting_rate, seed):
        """With no neurons firing, weights should not change."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, _, _ = create_test_inputs(n_neurons, seed)
        firing_prev = np.zeros(n_neurons, dtype=bool)
        firing_current = np.zeros(n_neurons, dtype=bool)

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert np.allclose(new_weights, weights)

    @given(
        n_neurons=n_neurons_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_no_ltp_with_zero_learning_rate(
        self, n_neurons, forgetting_rate, seed
    ):
        """With zero learning rate, LTP should not occur."""
        learner = HebbianLearner(
            learning_rate=0.0,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        rng = np.random.default_rng(seed)
        weights = np.ones((n_neurons, n_neurons)) * 0.5
        np.fill_diagonal(weights, 0)
        link_matrix = np.ones((n_neurons, n_neurons), dtype=bool)
        np.fill_diagonal(link_matrix, False)
        weights = np.where(link_matrix, weights, 0.0)

        # All neurons fire - should trigger LTP conditions
        firing_prev = np.ones(n_neurons, dtype=bool)
        firing_current = np.ones(n_neurons, dtype=bool)

        new_weights = learner.apply(weights, link_matrix, firing_prev, firing_current)

        # Weights should only decrease (LTD) or stay same, never increase
        assert np.all(new_weights <= weights)


class TestDeterminism:
    """Property tests for deterministic behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_same_input_same_output(
        self, n_neurons, learning_rate, forgetting_rate, seed
    ):
        """Same inputs must produce identical outputs."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )

        result1 = learner.apply(
            weights.copy(), link_matrix, firing_prev, firing_current
        )
        result2 = learner.apply(
            weights.copy(), link_matrix, firing_prev, firing_current
        )

        assert np.array_equal(result1, result2)

    @given(
        n_neurons=n_neurons_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_original_not_modified(
        self, n_neurons, learning_rate, forgetting_rate, seed
    ):
        """Original weight matrix must not be modified."""
        learner = HebbianLearner(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            weight_min=0.0,
            weight_max=1.0,
        )

        weights, link_matrix, firing_prev, firing_current = create_test_inputs(
            n_neurons, seed
        )
        original = weights.copy()

        _ = learner.apply(weights, link_matrix, firing_prev, firing_current)

        assert np.array_equal(weights, original)
