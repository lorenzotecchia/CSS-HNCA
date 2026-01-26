"""Unit tests for Network data structure.

Tests for Network.create_beta_weighted_directed factory method.
"""

import numpy as np
import pytest

from src.core.network import Network


class TestNetworkCreation:
    """Tests for Network.create_beta_weighted_directed factory method."""

    def test_create_returns_network(self):
        """create_beta_weighted_directed should return a Network instance."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.25,
            excitatory_fraction=1.0,
            seed=42,
        )
        assert isinstance(network, Network)

    def test_n_neurons_stored_correctly(self):
        """Network should store the number of neurons."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.25,
            excitatory_fraction=1.0,
            seed=42,
        )
        assert network.n_neurons == 50

    def test_box_size_is_unit_cube(self):
        """Network should have unit cube box size."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.25,
            seed=42,
        )
        assert network.box_size == (1.0, 1.0, 1.0)


class TestNetworkPositions:
    """Tests for neuron positions."""

    def test_positions_shape(self):
        """Positions should have shape (N, 3) for N neurons in 3D."""
        network = Network.create_beta_weighted_directed(
            n_neurons=25,
            k_prop=0.25,
            seed=42,
        )
        assert network.positions.shape == (25, 3)

    def test_positions_dtype(self):
        """Positions should be floating point."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.25,
            seed=42,
        )
        assert np.issubdtype(network.positions.dtype, np.floating)

    def test_positions_within_box_bounds(self):
        """All positions should be within [0, 1] for each dimension."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.25,
            seed=42,
        )
        assert np.all(network.positions >= 0)
        assert np.all(network.positions <= 1)


class TestNetworkLinkMatrix:
    """Tests for structural connectivity (link matrix)."""

    def test_link_matrix_shape(self):
        """Link matrix should have shape (N, N)."""
        network = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=42,
        )
        assert network.link_matrix.shape == (20, 20)

    def test_link_matrix_dtype_bool(self):
        """Link matrix should be boolean."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.25,
            seed=42,
        )
        assert network.link_matrix.dtype == np.bool_

    def test_link_matrix_no_self_connections(self):
        """Diagonal should be False (no self-connections)."""
        network = Network.create_beta_weighted_directed(
            n_neurons=15,
            k_prop=0.25,
            seed=42,
        )
        assert not np.any(np.diag(network.link_matrix))

    def test_link_matrix_directed(self):
        """Link matrix should be directed (contains cycle)."""
        # The beta-weighted network starts with a directed cycle
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.25,
            seed=42,
        )
        # Check cycle edges exist: i -> (i+1) % n
        for i in range(network.n_neurons):
            assert network.link_matrix[i, (i + 1) % network.n_neurons]

    def test_links_only_within_radius(self):
        """Links should only exist between neurons within radius distance."""
        network = Network.create_beta_weighted_directed(
            n_neurons=30,
            k_prop=0.25,
            seed=42,
        )
        # All links (except cycle) should be within the computed radius
        for i in range(network.n_neurons):
            for j in range(network.n_neurons):
                if i != j and network.link_matrix[i, j]:
                    # This is a valid link - either cycle or within radius
                    is_cycle = (i + 1) % network.n_neurons == j
                    if not is_cycle:
                        dist = np.linalg.norm(
                            network.positions[i] - network.positions[j]
                        )
                        assert dist <= network.radius


class TestNetworkWeightMatrix:
    """Tests for synaptic weight matrix."""

    def test_weight_matrix_shape(self):
        """Weight matrix should have shape (N, N)."""
        network = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=42,
        )
        assert network.weight_matrix.shape == (20, 20)

    def test_weight_matrix_dtype_float(self):
        """Weight matrix should be floating point."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.25,
            seed=42,
        )
        assert np.issubdtype(network.weight_matrix.dtype, np.floating)

    def test_weight_matrix_initial_values(self):
        """Weights should be beta-distributed where links exist."""
        network = Network.create_beta_weighted_directed(
            n_neurons=15,
            k_prop=0.25,
            excitatory_fraction=1.0,  # All excitatory
            seed=42,
        )
        linked_weights = network.weight_matrix[network.link_matrix]
        # Beta distribution produces values in (0, 1)
        assert np.all(linked_weights > 0)
        assert np.all(linked_weights <= 1)

    def test_weight_matrix_zero_where_no_link(self):
        """Weights should be 0 where there are no structural links."""
        network = Network.create_beta_weighted_directed(
            n_neurons=15,
            k_prop=0.25,
            seed=42,
        )
        non_linked_weights = network.weight_matrix[~network.link_matrix]
        assert np.all(non_linked_weights == 0)

    def test_weight_matrix_no_self_weights(self):
        """Diagonal should be 0 (no self-connections)."""
        network = Network.create_beta_weighted_directed(
            n_neurons=15,
            k_prop=0.25,
            seed=42,
        )
        assert np.all(np.diag(network.weight_matrix) == 0)


class TestNeuronTypes:
    """Tests for excitatory/inhibitory neuron differentiation."""

    def test_network_has_neuron_types(self):
        """Network should have neuron_types array."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.25,
            seed=42,
        )
        assert hasattr(network, "neuron_types")
        assert network.neuron_types.shape == (100,)
        assert network.neuron_types.dtype == bool

    def test_excitatory_fraction_respected(self):
        """Approximately excitatory_fraction of neurons should be excitatory."""
        network = Network.create_beta_weighted_directed(
            n_neurons=1000,
            k_prop=0.25,
            excitatory_fraction=0.8,
            seed=42,
        )
        actual_fraction = np.mean(network.neuron_types)
        assert 0.75 < actual_fraction < 0.85

    def test_excitatory_neurons_have_positive_weights(self):
        """Excitatory neurons should have positive outgoing weights."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.1,
            excitatory_fraction=0.5,
            seed=42,
        )
        for i in range(network.n_neurons):
            if network.neuron_types[i]:
                row_weights = network.weight_matrix[i, network.link_matrix[i]]
                if len(row_weights) > 0:
                    assert np.all(row_weights >= 0)

    def test_inhibitory_neurons_have_negative_weights(self):
        """Inhibitory neurons should have negative outgoing weights."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.1,
            excitatory_fraction=0.5,
            seed=42,
        )
        for i in range(network.n_neurons):
            if not network.neuron_types[i]:
                row_weights = network.weight_matrix[i, network.link_matrix[i]]
                if len(row_weights) > 0:
                    assert np.all(row_weights <= 0)

    def test_all_inhibitory_when_fraction_zero(self):
        """All neurons should be inhibitory when excitatory_fraction=0."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.25,
            excitatory_fraction=0.0,
            seed=42,
        )
        assert not np.any(network.neuron_types)  # All False

    def test_all_excitatory_when_fraction_one(self):
        """All neurons should be excitatory when excitatory_fraction=1."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.25,
            excitatory_fraction=1.0,
            seed=42,
        )
        assert np.all(network.neuron_types)  # All True


class TestNetworkReproducibility:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_same_positions(self):
        """Same seed should produce identical positions."""
        net1 = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=12345,
        )
        net2 = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=12345,
        )
        assert np.array_equal(net1.positions, net2.positions)

    def test_different_seed_different_positions(self):
        """Different seeds should produce different positions."""
        net1 = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=111,
        )
        net2 = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.25,
            seed=222,
        )
        assert not np.array_equal(net1.positions, net2.positions)


class TestInhibitoryNodes:
    """Tests for inhibitory neuron functionality."""

    def test_excitatory_fraction_one_no_inhibitory(self):
        """With excitatory_fraction=1, no neurons should be inhibitory."""
        network = Network.create_beta_weighted_directed(
            n_neurons=100,
            k_prop=0.1,
            excitatory_fraction=1.0,
            seed=42,
        )
        assert not np.any(network.inhibitory_nodes)

    def test_excitatory_fraction_zero_all_inhibitory(self):
        """With excitatory_fraction=0, all neurons should be inhibitory."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.1,
            excitatory_fraction=0.0,
            seed=42,
        )
        assert np.all(network.inhibitory_nodes)

    def test_inhibitory_proportion_correct_count(self):
        """Proportion of inhibitory neurons should match 1 - excitatory_fraction."""
        n_neurons = 200
        exc_frac = 0.7
        network = Network.create_beta_weighted_directed(
            n_neurons=n_neurons,
            k_prop=0.1,
            excitatory_fraction=exc_frac,
            seed=42,
        )
        actual_inh_prop = np.mean(network.inhibitory_nodes)
        expected_inh_prop = 1.0 - exc_frac
        # Allow some tolerance due to randomness
        assert abs(actual_inh_prop - expected_inh_prop) < 0.1

    def test_inhibitory_out_degrees_negative(self):
        """All out-degrees of inhibitory neurons should have negative weights."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.2,
            excitatory_fraction=0.5,
            seed=42,
        )
        for i in range(network.n_neurons):
            if network.inhibitory_nodes[i]:
                out_weights = network.weight_matrix[i, network.link_matrix[i, :]]
                if len(out_weights) > 0:
                    assert np.all(out_weights < 0), f"Inhibitory neuron {i} has positive out-weights"

    def test_excitatory_out_degrees_positive(self):
        """All out-degrees of excitatory neurons should have positive weights."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.2,
            excitatory_fraction=0.5,
            seed=42,
        )
        for i in range(network.n_neurons):
            if not network.inhibitory_nodes[i]:
                out_weights = network.weight_matrix[i, network.link_matrix[i, :]]
                if len(out_weights) > 0:
                    assert np.all(out_weights > 0), f"Excitatory neuron {i} has negative out-weights"

    def test_inhibitory_weights_bounds(self):
        """Inhibitory weights should be bounded by weight_min_inh and weight_max_inh."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.2,
            excitatory_fraction=0.5,
            weight_min_inh=-1.0,
            weight_max_inh=0.0,
            seed=42,
        )
        for i in range(network.n_neurons):
            if network.inhibitory_nodes[i]:
                out_weights = network.weight_matrix[i, network.link_matrix[i, :]]
                if len(out_weights) > 0:
                    assert np.all(out_weights >= -1.0) and np.all(out_weights <= 0)

    def test_excitatory_weights_bounds(self):
        """Excitatory weights should be bounded by weight_min and weight_max."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.2,
            excitatory_fraction=0.5,
            weight_min=0.0,
            weight_max=1.0,
            seed=42,
        )
        for i in range(network.n_neurons):
            if not network.inhibitory_nodes[i]:
                out_weights = network.weight_matrix[i, network.link_matrix[i, :]]
                if len(out_weights) > 0:
                    assert np.all(out_weights >= 0) and np.all(out_weights <= 1.0)

    def test_excitatory_fraction_validation(self):
        """excitatory_fraction should be validated."""
        with pytest.raises(ValueError, match="excitatory_fraction must be in"):
            Network.create_beta_weighted_directed(
                n_neurons=10,
                k_prop=0.25,
                excitatory_fraction=-0.1,
            )
        with pytest.raises(ValueError, match="excitatory_fraction must be in"):
            Network.create_beta_weighted_directed(
                n_neurons=10,
                k_prop=0.25,
                excitatory_fraction=1.1,
            )
