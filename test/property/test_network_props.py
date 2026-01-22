"""Property-based tests for Network using Hypothesis.

RED phase: These tests should fail until Network is implemented.
"""

import numpy as np
from hypothesis import given, strategies as st, assume, settings

from src.core.network import Network


# Custom strategies for network parameters
n_neurons_strategy = st.integers(min_value=2, max_value=100)
box_dim_strategy = st.floats(min_value=1.0, max_value=100.0, allow_nan=False)
radius_strategy = st.floats(min_value=0.1, max_value=50.0, allow_nan=False)
weight_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


class TestNetworkPositionProperties:
    """Property tests for neuron positions."""

    @given(
        n_neurons=n_neurons_strategy,
        box_x=box_dim_strategy,
        box_y=box_dim_strategy,
        box_z=box_dim_strategy,
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_positions_always_within_bounds(
        self, n_neurons, box_x, box_y, box_z, radius, initial_weight, seed
    ):
        """Positions must always be within [0, box_size] for all dimensions."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=(box_x, box_y, box_z),
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert np.all(network.positions >= 0)
        assert np.all(network.positions[:, 0] <= box_x)
        assert np.all(network.positions[:, 1] <= box_y)
        assert np.all(network.positions[:, 2] <= box_z)

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_positions_shape_invariant(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Positions shape must always be (n_neurons, 3)."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert network.positions.shape == (n_neurons, 3)


class TestNetworkLinkMatrixProperties:
    """Property tests for link matrix."""

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_link_matrix_shape_invariant(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Link matrix shape must always be (n_neurons, n_neurons)."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert network.link_matrix.shape == (n_neurons, n_neurons)

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_no_self_connections_invariant(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Diagonal of link matrix must always be False (no self-loops)."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        diagonal = np.diag(network.link_matrix)
        assert not np.any(diagonal)

    @given(
        n_neurons=st.integers(min_value=2, max_value=50),
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_link_matrix_matches_distance(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Links must exist iff distance <= radius."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j:
                    dist = np.linalg.norm(
                        network.positions[i] - network.positions[j]
                    )
                    if network.link_matrix[i, j]:
                        assert dist <= radius, (
                            f"Link exists but distance {dist} > radius {radius}"
                        )
                    else:
                        assert dist > radius, (
                            f"No link but distance {dist} <= radius {radius}"
                        )

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_link_matrix_symmetric(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Link matrix must be symmetric (distance is symmetric)."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert np.array_equal(network.link_matrix, network.link_matrix.T)


class TestNetworkWeightMatrixProperties:
    """Property tests for weight matrix."""

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_weight_matrix_shape_invariant(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Weight matrix shape must always be (n_neurons, n_neurons)."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert network.weight_matrix.shape == (n_neurons, n_neurons)

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_weights_zero_where_no_link(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Weights must be zero where no structural link exists."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        # Where link_matrix is False, weight_matrix must be 0
        no_link_weights = network.weight_matrix[~network.link_matrix]
        assert np.all(no_link_weights == 0)

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_weights_initialized_where_link(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Weights must equal initial_weight where link exists."""
        network = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        # Where link_matrix is True, weight_matrix should be initial_weight
        if np.any(network.link_matrix):
            linked_weights = network.weight_matrix[network.link_matrix]
            assert np.allclose(linked_weights, initial_weight)


class TestNetworkReproducibilityProperties:
    """Property tests for reproducibility."""

    @given(
        n_neurons=n_neurons_strategy,
        box_size=st.tuples(box_dim_strategy, box_dim_strategy, box_dim_strategy),
        radius=radius_strategy,
        initial_weight=weight_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_same_seed_produces_identical_network(
        self, n_neurons, box_size, radius, initial_weight, seed
    ):
        """Same parameters and seed must produce identical networks."""
        net1 = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )
        net2 = Network.create_random(
            n_neurons=n_neurons,
            box_size=box_size,
            radius=radius,
            initial_weight=initial_weight,
            seed=seed,
        )

        assert np.array_equal(net1.positions, net2.positions)
        assert np.array_equal(net1.link_matrix, net2.link_matrix)
        assert np.array_equal(net1.weight_matrix, net2.weight_matrix)
