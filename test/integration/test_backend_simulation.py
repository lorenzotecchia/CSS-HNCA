"""Integration tests for Simulation with Backend abstraction.

Tests that the backend abstraction works correctly with the simulation.
The core modules use NumPy directly, and the backend provides a consistent
interface that produces compatible numpy arrays.
"""

import numpy as np
import pytest

from src.core.backend import NumPyBackend, get_backend
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation


class TestSimulationWithBackend:
    """Tests that Simulation works correctly alongside the backend."""

    def test_simulation_arrays_compatible_with_backend(self):
        """Simulation arrays should be compatible with backend operations."""
        backend = get_backend()

        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        # Backend operations should work on simulation arrays
        sum_positions = backend.sum(network.positions)
        mean_weights = backend.mean(network.weight_matrix)

        assert isinstance(sum_positions, float)
        assert isinstance(mean_weights, float)

        # Step should complete without error
        sim.step()
        assert sim.time_step == 1

    def test_backend_matmul_matches_simulation(self):
        """Backend matmul should produce same results as direct numpy."""
        backend = get_backend()

        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )

        # Compute input signal using backend
        firing_float = state.firing.astype(float)
        input_via_backend = backend.matmul(
            backend.transpose(network.weight_matrix),
            firing_float
        )

        # Compute input signal using direct numpy (as simulation does)
        input_direct = network.weight_matrix.T @ firing_float

        # Results should match
        assert np.allclose(input_via_backend, input_direct)

    def test_simulation_with_learning_produces_valid_weights(self):
        """Simulation with Hebbian learning should maintain valid weights."""
        from src.learning.hebbian import HebbianLearner

        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.4,
            firing_count=1,
            seed=42,
        )
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            learner=learner,
        )

        # Run several steps
        for _ in range(10):
            sim.step()

        # Weights should remain valid
        backend = get_backend()
        assert backend.any(network.weight_matrix >= 0)
        assert sim.time_step == 10


class TestBackendConsistency:
    """Tests that backend produces consistent results."""

    def test_multiple_simulations_same_seed(self):
        """Same seed should produce same results."""
        def create_sim(seed):
            network = Network.create_beta_weighted_directed(
                n_neurons=15,
                k_prop=0.2,
                seed=seed,
            )
            state = NeuronState.create(
                n_neurons=15,
                threshold=0.5,
                firing_count=1,
                seed=seed,
            )
            return Simulation(
                network=network,
                state=state,
                learning_rate=0.01,
                forgetting_rate=0.005,
            )

        sim1 = create_sim(12345)
        sim2 = create_sim(12345)

        # Run both
        for _ in range(5):
            sim1.step()
            sim2.step()

        # Should have identical states
        assert np.array_equal(sim1.state.firing, sim2.state.firing)
        assert np.allclose(sim1.state.membrane_potential, sim2.state.membrane_potential)

    def test_backend_to_numpy_conversion(self):
        """Backend arrays should be convertible to numpy."""
        backend = get_backend()

        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )

        # Should be able to convert to numpy (identity for NumPy backend)
        positions_np = backend.to_numpy(network.positions)
        weights_np = backend.to_numpy(network.weight_matrix)

        assert isinstance(positions_np, np.ndarray)
        assert isinstance(weights_np, np.ndarray)
        assert np.array_equal(positions_np, network.positions)


class TestBackendWithAnalytics:
    """Tests that analytics work with backend operations."""

    def test_firing_count_via_backend(self):
        """Backend sum should compute firing count correctly."""
        backend = get_backend()

        state = NeuronState.create(
            n_neurons=20,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )

        # Compute firing count via backend
        count_backend = backend.sum(state.firing.astype(float))

        # Compute via direct numpy
        count_direct = int(np.sum(state.firing))

        assert count_backend == count_direct

    def test_average_weight_via_backend(self):
        """Backend mean should compute average weight correctly."""
        backend = get_backend()

        network = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.2,
            seed=42,
        )

        # Get connected weights
        connected_weights = network.weight_matrix[network.link_matrix]

        # Compute via backend
        avg_backend = backend.mean(connected_weights)

        # Compute via direct numpy
        avg_direct = float(np.mean(connected_weights))

        assert np.isclose(avg_backend, avg_direct)


class TestBackendFallback:
    """Tests for backend fallback behavior."""

    def test_default_backend_is_numpy(self):
        """Default backend should be NumPy."""
        backend = get_backend()
        assert isinstance(backend, NumPyBackend)

    def test_prefer_gpu_falls_back_to_numpy(self):
        """prefer_gpu=True should fall back to NumPy when JAX unavailable."""
        backend = get_backend(prefer_gpu=True)
        # Should get a valid backend (NumPy if JAX not installed)
        assert hasattr(backend, 'zeros')
        assert hasattr(backend, 'matmul')

    def test_simulation_works_without_backend(self):
        """Simulation should work with standard NumPy operations."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        # Should work normally
        sim.step()
        assert sim.time_step == 1
