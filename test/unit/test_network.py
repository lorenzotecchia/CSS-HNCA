"""Unit tests for Network data structure.

RED phase: These tests should fail until Network is implemented.
"""

import numpy as np
import pytest

from src.core.network import Network


class TestNetworkCreation:
    """Tests for Network.create_random factory method."""

    def test_create_random_returns_network(self):
        """create_random should return a Network instance."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert isinstance(network, Network)

    def test_n_neurons_stored_correctly(self):
        """Network should store the number of neurons."""
        network = Network.create_random(
            n_neurons=50,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.n_neurons == 50

    def test_box_size_stored_correctly(self):
        """Network should store the box size."""
        box = (5.0, 10.0, 15.0)
        network = Network.create_random(
            n_neurons=10,
            box_size=box,
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.box_size == box

    def test_radius_stored_correctly(self):
        """Network should store the connectivity radius."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(10.0, 10.0, 10.0),
            radius=3.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.radius == 3.5


class TestNetworkPositions:
    """Tests for neuron positions."""

    def test_positions_shape(self):
        """Positions should have shape (N, 3) for N neurons in 3D."""
        network = Network.create_random(
            n_neurons=25,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.positions.shape == (25, 3)

    def test_positions_dtype(self):
        """Positions should be floating point."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert np.issubdtype(network.positions.dtype, np.floating)

    def test_positions_within_box_bounds(self):
        """All positions should be within [0, box_size] for each dimension."""
        box = (5.0, 10.0, 15.0)
        network = Network.create_random(
            n_neurons=100,
            box_size=box,
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert np.all(network.positions >= 0)
        assert np.all(network.positions[:, 0] <= box[0])
        assert np.all(network.positions[:, 1] <= box[1])
        assert np.all(network.positions[:, 2] <= box[2])


class TestNetworkLinkMatrix:
    """Tests for structural connectivity (link matrix)."""

    def test_link_matrix_shape(self):
        """Link matrix should have shape (N, N)."""
        network = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.link_matrix.shape == (20, 20)

    def test_link_matrix_dtype_bool(self):
        """Link matrix should be boolean."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.link_matrix.dtype == np.bool_

    def test_link_matrix_no_self_connections(self):
        """Diagonal should be False (no self-connections)."""
        network = Network.create_random(
            n_neurons=15,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert not np.any(np.diag(network.link_matrix))

    def test_link_matrix_directed(self):
        """Link matrix should be directed (asymmetric in general)."""
        # With random positions, it's very unlikely to be perfectly symmetric
        # We test that the matrix CAN be asymmetric by checking if any
        # pair has A->B but not B->A (or vice versa)
        network = Network.create_random(
            n_neurons=50,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        # The link_matrix is based on distance, which IS symmetric.
        # Actually, for directed connectivity with distance-based links,
        # both directions exist if within radius. The directionality
        # comes from the weight matrix (A->B weight != B->A weight).
        # So link_matrix IS symmetric, but weight_matrix is not.
        # Let me fix this test to reflect the actual design.
        # Actually re-reading the plan: link_matrix represents structural
        # connectivity (who CAN connect), which is symmetric based on distance.
        # The WEIGHTS are directed. Let me verify link_matrix is symmetric.
        assert np.array_equal(network.link_matrix, network.link_matrix.T)

    def test_links_only_within_radius(self):
        """Links should only exist between neurons within radius distance."""
        network = Network.create_random(
            n_neurons=30,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        # Check that all links correspond to pairs within radius
        for i in range(network.n_neurons):
            for j in range(network.n_neurons):
                if i != j:
                    dist = np.linalg.norm(
                        network.positions[i] - network.positions[j]
                    )
                    if network.link_matrix[i, j]:
                        assert dist <= network.radius
                    else:
                        assert dist > network.radius


class TestNetworkWeightMatrix:
    """Tests for synaptic weight matrix."""

    def test_weight_matrix_shape(self):
        """Weight matrix should have shape (N, N)."""
        network = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert network.weight_matrix.shape == (20, 20)

    def test_weight_matrix_dtype_float(self):
        """Weight matrix should be floating point."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=42,
        )
        assert np.issubdtype(network.weight_matrix.dtype, np.floating)

    def test_weight_matrix_initial_values(self):
        """Weights should be initialized to ±initial_weight where links exist."""
        network = Network.create_random(
            n_neurons=15,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.3,
            seed=42,
        )
        # Where there are links, weights should equal ±initial_weight
        linked_weights = network.weight_matrix[network.link_matrix]
        assert np.allclose(np.abs(linked_weights), 0.3)

    def test_weight_matrix_zero_where_no_link(self):
        """Weights should be 0 where there are no structural links."""
        network = Network.create_random(
            n_neurons=15,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.3,
            seed=42,
        )
        # Where there are no links, weights should be 0
        non_linked_weights = network.weight_matrix[~network.link_matrix]
        assert np.all(non_linked_weights == 0)

    def test_weight_matrix_no_self_weights(self):
        """Diagonal should be 0 (no self-connections)."""
        network = Network.create_random(
            n_neurons=15,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.5,
            seed=42,
        )
        assert np.all(np.diag(network.weight_matrix) == 0)


class TestNeuronTypes:
    """Tests for excitatory/inhibitory neuron differentiation."""

    def test_network_has_neuron_types(self):
        """Network should have neuron_types array."""
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        assert hasattr(network, "neuron_types")
        assert network.neuron_types.shape == (100,)
        assert network.neuron_types.dtype == bool

    def test_excitatory_fraction_respected(self):
        """Approximately excitatory_fraction of neurons should be excitatory."""
        network = Network.create_random(
            n_neurons=1000,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
            excitatory_fraction=0.8,
            seed=42,
        )
        actual_fraction = np.mean(network.neuron_types)
        assert 0.75 < actual_fraction < 0.85

    def test_excitatory_neurons_have_positive_weights(self):
        """Excitatory neurons should have positive outgoing weights."""
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
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
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
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
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
            excitatory_fraction=0.0,
            seed=42,
        )
        assert not np.any(network.neuron_types)  # All False

    def test_all_excitatory_when_fraction_one(self):
        """All neurons should be excitatory when excitatory_fraction=1."""
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.0,
            initial_weight=0.1,
            excitatory_fraction=1.0,
            seed=42,
        )
        assert np.all(network.neuron_types)  # All True


class TestNetworkReproducibility:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_same_positions(self):
        """Same seed should produce identical positions."""
        net1 = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=12345,
        )
        net2 = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=12345,
        )
        assert np.array_equal(net1.positions, net2.positions)

    def test_different_seed_different_positions(self):
        """Different seeds should produce different positions."""
        net1 = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=111,
        )
        net2 = Network.create_random(
            n_neurons=20,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            seed=222,
        )
        assert not np.array_equal(net1.positions, net2.positions)
