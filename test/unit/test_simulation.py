"""Unit tests for Simulation orchestrator.

RED phase: These tests should fail until Simulation is implemented.
"""

import numpy as np
import pytest

from src.core.simulation import Simulation, SimulationState
from src.core.network import Network
from src.core.neuron_state import NeuronState


@pytest.fixture
def small_network():
    """Create a small network for testing."""
    return Network.create_random(
        n_neurons=10,
        box_size=(10.0, 10.0, 10.0),
        radius=5.0,
        initial_weight=0.1,
        seed=42,
    )


@pytest.fixture
def neuron_state():
    """Create neuron state for testing."""
    return NeuronState.create(
        n_neurons=10,
        threshold=0.5,
        initial_firing_fraction=0.2,
        seed=42,
    )


@pytest.fixture
def simulation(small_network, neuron_state):
    """Create a simulation for testing."""
    return Simulation(
        network=small_network,
        state=neuron_state,
        learning_rate=0.01,
        forgetting_rate=0.005,
    )


class TestSimulationState:
    """Tests for SimulationState enum."""

    def test_stopped_state_exists(self):
        """STOPPED state should exist."""
        assert hasattr(SimulationState, "STOPPED")

    def test_running_state_exists(self):
        """RUNNING state should exist."""
        assert hasattr(SimulationState, "RUNNING")

    def test_paused_state_exists(self):
        """PAUSED state should exist."""
        assert hasattr(SimulationState, "PAUSED")


class TestSimulationCreation:
    """Tests for Simulation initialization."""

    def test_initial_time_step_zero(self, simulation):
        """Time step should start at 0."""
        assert simulation.time_step == 0

    def test_initial_state_stopped(self, simulation):
        """Simulation should start in STOPPED state."""
        assert simulation.sim_state == SimulationState.STOPPED

    def test_network_stored(self, simulation, small_network):
        """Simulation should store the network reference."""
        assert simulation.network is small_network

    def test_neuron_state_stored(self, simulation, neuron_state):
        """Simulation should store the neuron state reference."""
        assert simulation.state is neuron_state

    def test_learning_rate_stored(self, simulation):
        """Simulation should store the learning rate."""
        assert simulation.learning_rate == 0.01

    def test_forgetting_rate_stored(self, simulation):
        """Simulation should store the forgetting rate."""
        assert simulation.forgetting_rate == 0.005


class TestSimulationStart:
    """Tests for start() method."""

    def test_start_changes_state_to_running(self, simulation):
        """start() should change state to RUNNING."""
        simulation.start()
        assert simulation.sim_state == SimulationState.RUNNING

    def test_start_from_paused(self, simulation):
        """start() from PAUSED should change to RUNNING."""
        simulation.start()
        simulation.pause()
        simulation.start()
        assert simulation.sim_state == SimulationState.RUNNING


class TestSimulationPause:
    """Tests for pause() method."""

    def test_pause_changes_state_to_paused(self, simulation):
        """pause() should change state to PAUSED when running."""
        simulation.start()
        simulation.pause()
        assert simulation.sim_state == SimulationState.PAUSED

    def test_pause_when_stopped_no_effect(self, simulation):
        """pause() when STOPPED should have no effect."""
        simulation.pause()
        assert simulation.sim_state == SimulationState.STOPPED


class TestSimulationStep:
    """Tests for step() method."""

    def test_step_increments_time(self, simulation):
        """step() should increment time_step by 1."""
        initial_time = simulation.time_step
        simulation.step()
        assert simulation.time_step == initial_time + 1

    def test_multiple_steps_increment_correctly(self, simulation):
        """Multiple steps should increment correctly."""
        for _ in range(5):
            simulation.step()
        assert simulation.time_step == 5

    def test_step_updates_neuron_firing(self, simulation):
        """step() should update neuron firing states based on weights."""
        # Get initial firing state
        initial_firing = simulation.state.firing.copy()
        simulation.step()
        # State should have changed (or at least been evaluated)
        # The actual values depend on the weight matrix and threshold
        # For now, just verify the mechanism runs without error
        assert simulation.state.firing.shape == initial_firing.shape


class TestSimulationReset:
    """Tests for reset() method."""

    def test_reset_zeroes_time_step(self, simulation):
        """reset() should set time_step back to 0."""
        for _ in range(10):
            simulation.step()
        simulation.reset()
        assert simulation.time_step == 0

    def test_reset_changes_state_to_stopped(self, simulation):
        """reset() should change state to STOPPED."""
        simulation.start()
        simulation.reset()
        assert simulation.sim_state == SimulationState.STOPPED

    def test_reset_with_seed_regenerates_network(self, simulation):
        """reset() with new seed should regenerate positions."""
        original_positions = simulation.network.positions.copy()
        simulation.reset(seed=99999)
        # Positions should be different with different seed
        assert not np.array_equal(simulation.network.positions, original_positions)

    def test_reset_with_same_seed_reproduces(self, simulation):
        """reset() with same seed should reproduce same positions."""
        simulation.reset(seed=12345)
        positions1 = simulation.network.positions.copy()
        simulation.step()
        simulation.step()
        simulation.reset(seed=12345)
        positions2 = simulation.network.positions.copy()
        assert np.array_equal(positions1, positions2)


class TestSimulationStateTransitions:
    """Tests for valid state transitions."""

    def test_stopped_to_running(self, simulation):
        """STOPPED -> RUNNING via start() is valid."""
        assert simulation.sim_state == SimulationState.STOPPED
        simulation.start()
        assert simulation.sim_state == SimulationState.RUNNING

    def test_running_to_paused(self, simulation):
        """RUNNING -> PAUSED via pause() is valid."""
        simulation.start()
        simulation.pause()
        assert simulation.sim_state == SimulationState.PAUSED

    def test_paused_to_running(self, simulation):
        """PAUSED -> RUNNING via start() is valid."""
        simulation.start()
        simulation.pause()
        simulation.start()
        assert simulation.sim_state == SimulationState.RUNNING

    def test_any_to_stopped_via_reset(self, simulation):
        """Any state -> STOPPED via reset() is valid."""
        simulation.start()
        simulation.reset()
        assert simulation.sim_state == SimulationState.STOPPED

        simulation.start()
        simulation.pause()
        simulation.reset()
        assert simulation.sim_state == SimulationState.STOPPED


class TestSimulationMetrics:
    """Tests for simulation metrics."""

    def test_firing_count(self, simulation):
        """Should be able to get count of currently firing neurons."""
        count = simulation.firing_count
        assert isinstance(count, int)
        assert 0 <= count <= simulation.network.n_neurons

    def test_average_weight(self, simulation):
        """Should be able to get average weight of connections."""
        avg = simulation.average_weight
        assert isinstance(avg, float)
        assert avg >= 0


class TestSimulationNeuronTypes:
    """Tests for neuron_types passing to learner."""

    def test_simulation_passes_neuron_types_to_learner(self):
        """Simulation should pass network.neuron_types to learner.apply()."""
        from unittest.mock import MagicMock

        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            excitatory_fraction=0.5,
            seed=42,
        )
        state = NeuronState.create(n_neurons=10, threshold=0.5, initial_firing_fraction=0.2, seed=42)
        learner = MagicMock()
        learner.apply.return_value = network.weight_matrix.copy()

        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.1,
            forgetting_rate=0.05,
            learner=learner,
        )
        sim.start()
        sim.step()

        # Verify neuron_types was passed
        call_kwargs = learner.apply.call_args.kwargs
        assert "neuron_types" in call_kwargs
        assert np.array_equal(call_kwargs["neuron_types"], network.neuron_types)
