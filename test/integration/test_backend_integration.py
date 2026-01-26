"""Integration tests for backend parameter propagation."""

import numpy as np
import pytest

from src.core.backend import NumPyBackend, get_backend
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner


class TestBackendIntegration:
    """Test that all components work with explicit backend."""

    def test_simulation_with_explicit_numpy_backend(self):
        """Verify all components use the same backend."""
        backend = NumPyBackend()

        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
            backend=backend,
        )

        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=1,
            seed=42,
            backend=backend,
        )

        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.01,
            weight_min=0.0,
            weight_max=1.0,
            backend=backend,
        )

        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.01,
            learner=learner,
            backend=backend,
        )

        sim.start()
        for _ in range(10):
            sim.step()

        assert sim.time_step == 10
        assert 0 <= sim.firing_count <= 10

    def test_backward_compatibility_no_backend(self):
        """Verify components work without explicit backend."""
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

        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.01,
            weight_min=0.0,
            weight_max=1.0,
        )

        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.01,
            learner=learner,
        )

        sim.start()
        sim.step()

        assert sim.time_step == 1

    def test_get_backend_returns_numpy_by_default(self):
        """Verify get_backend() returns NumPyBackend by default."""
        backend = get_backend()
        assert isinstance(backend, NumPyBackend)

    def test_get_backend_prefer_gpu_fallback(self):
        """Verify get_backend(prefer_gpu=True) falls back to NumPy when JAX unavailable."""
        backend = get_backend(prefer_gpu=True)
        # Should still work (returns NumPy if JAX not installed)
        assert backend is not None
        arr = backend.zeros((5,))
        assert arr.shape == (5,)
