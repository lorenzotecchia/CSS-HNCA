"""Property-based tests for Network using Hypothesis.

Tests Network invariants using the beta-weighted directed network.
"""

import numpy as np
from hypothesis import given, strategies as st, settings

from src.core.network import Network


# Custom strategies for network parameters
n_neurons_strategy = st.integers(min_value=5, max_value=50)
k_prop_strategy = st.floats(min_value=0.1, max_value=0.5, allow_nan=False)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)
excitatory_fraction_strategy = st.floats(min_value=0.0, max_value=1.0)


def create_network(
    n_neurons: int,
    k_prop: float | None = None,
    excitatory_fraction: float = 0.8,
    seed: int = 42,
) -> Network:
    """Helper to create a network for testing."""
    # Compute valid k_prop range for given n_neurons (must be in [2/n, 1-1/n])
    k_min = 2 / n_neurons
    k_max = 1 - 1 / n_neurons
    if k_prop is None or k_prop < k_min or k_prop > k_max:
        k_prop = (k_min + k_max) / 2  # Use midpoint of valid range
    return Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=k_prop,
        excitatory_fraction=excitatory_fraction,
        seed=seed,
    )


class TestNetworkPositionProperties:
    """Property tests for neuron positions."""

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_positions_always_within_bounds(self, n_neurons, k_prop, seed):
        """Positions must always be within [0, 1] for unit cube."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        assert np.all(network.positions >= 0)
        assert np.all(network.positions <= 1)

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_positions_shape_invariant(self, n_neurons, k_prop, seed):
        """Positions shape must always be (n_neurons, 3)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        assert network.positions.shape == (n_neurons, 3)


class TestNetworkLinkMatrixProperties:
    """Property tests for link matrix."""

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_link_matrix_shape_invariant(self, n_neurons, k_prop, seed):
        """Link matrix shape must always be (n_neurons, n_neurons)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        assert network.link_matrix.shape == (n_neurons, n_neurons)

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_no_self_connections_invariant(self, n_neurons, k_prop, seed):
        """Diagonal of link matrix must always be False (no self-loops)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        diagonal = np.diag(network.link_matrix)
        assert not np.any(diagonal)

    @given(
        n_neurons=st.integers(min_value=5, max_value=30),
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_link_matrix_has_required_edges(self, n_neurons, k_prop, seed):
        """Network has at least n_neurons edges (directed cycle)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        # Directed network includes a cycle (n edges) plus additional edges
        total_edges = np.sum(network.link_matrix)
        assert total_edges >= n_neurons, (
            f"Expected at least {n_neurons} edges (cycle), got {total_edges}"
        )

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_link_matrix_directed(self, n_neurons, k_prop, seed):
        """Directed network link matrix may have asymmetric connections."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        # Directed networks can have asymmetric links
        # Verify shape is correct
        assert network.link_matrix.shape == (n_neurons, n_neurons)


class TestNetworkWeightMatrixProperties:
    """Property tests for weight matrix."""

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_weight_matrix_shape_invariant(self, n_neurons, k_prop, seed):
        """Weight matrix shape must always be (n_neurons, n_neurons)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        assert network.weight_matrix.shape == (n_neurons, n_neurons)

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_weights_zero_where_no_link(self, n_neurons, k_prop, seed):
        """Weights must be zero where no structural link exists."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        # Where link_matrix is False, weight_matrix must be 0
        no_link_weights = network.weight_matrix[~network.link_matrix]
        assert np.all(no_link_weights == 0)

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_weights_initialized_where_link(self, n_neurons, k_prop, seed):
        """Weights must be non-zero where link exists (beta distribution)."""
        network = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        # Where link_matrix is True, weight_matrix should be non-zero
        if np.any(network.link_matrix):
            linked_weights = network.weight_matrix[network.link_matrix]
            assert np.all(linked_weights != 0)


class TestNetworkReproducibilityProperties:
    """Property tests for reproducibility."""

    @given(
        n_neurons=n_neurons_strategy,
        k_prop=k_prop_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_same_seed_produces_identical_network(self, n_neurons, k_prop, seed):
        """Same parameters and seed must produce identical networks."""
        net1 = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)
        net2 = create_network(n_neurons=n_neurons, k_prop=k_prop, seed=seed)

        assert np.array_equal(net1.positions, net2.positions)
        assert np.array_equal(net1.link_matrix, net2.link_matrix)
        assert np.array_equal(net1.weight_matrix, net2.weight_matrix)


class TestNeuronTypeProperties:
    """Property tests for neuron type invariants."""

    @given(
        n_neurons=st.integers(min_value=10, max_value=50),
        excitatory_fraction=excitatory_fraction_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=50)
    def test_neuron_types_length_matches_n_neurons(
        self, n_neurons, excitatory_fraction, seed
    ):
        """neuron_types array should have length n_neurons."""
        network = create_network(
            n_neurons=n_neurons,
            excitatory_fraction=excitatory_fraction,
            seed=seed,
        )
        assert len(network.neuron_types) == n_neurons

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(deadline=None, max_examples=50)
    def test_weight_signs_match_neuron_types(self, seed):
        """Excitatory neurons have non-negative weights, inhibitory non-positive."""
        network = create_network(
            n_neurons=30,
            excitatory_fraction=0.5,
            seed=seed,
        )
        for i in range(network.n_neurons):
            connected = network.link_matrix[i]
            if np.any(connected):
                weights = network.weight_matrix[i, connected]
                if network.neuron_types[i]:  # Excitatory
                    assert np.all(weights >= 0)
                else:  # Inhibitory
                    assert np.all(weights <= 0)
