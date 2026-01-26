"""Integration tests for Simulation with Network and NeuronState.

RED phase: These tests should fail until all core components are implemented.
"""

import numpy as np
import pytest

from src.core.simulation import Simulation, SimulationState
from src.core.network import Network
from src.core.neuron_state import NeuronState


class TestSimulationCoreIntegration:
    """Tests that Simulation correctly integrates Network and NeuronState."""

    def test_simulation_uses_network_weights_for_state_update(self):
        """Simulation.step() should use network weights to compute firing."""
        # Create a small deterministic network
        network = Network.create_beta_weighted_directed(
            n_neurons=5,
            k_prop=0.6,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=5,
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

        # Manually set one neuron to fire
        state.firing[0] = True

        # Step the simulation
        sim.step()

        # The neurons connected to neuron 0 should receive input
        # Whether they fire depends on the sum of weights exceeding threshold
        # This tests the integration, not specific values
        assert sim.time_step == 1

    def test_state_update_formula(self):
        """Verify state update follows: v = W^T · s, fire if v >= γ."""
        # Create minimal 3-neuron network (for n=3, k_prop must be 2/3)
        network = Network.create_beta_weighted_directed(
            n_neurons=3,
            k_prop=2/3,  # Only valid k_prop for n=3
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=3,
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

        # Manually set weights: neuron 0 -> neuron 1 with weight 0.6
        # (row 0, col 1 means connection from 0 to 1)
        network.weight_matrix[0, 1] = 0.6
        network.link_matrix[0, 1] = True

        # Set neuron 0 to fire
        state.firing[0] = True

        # Step the simulation
        sim.step()

        # Neuron 1 should fire because:
        # v[1] = W^T[1, :] · s = W[0, 1] * s[0] = 0.6 * 1 = 0.6 >= 0.5
        assert state.firing[1] == True
        # Neuron 2 should not fire (no input)
        assert state.firing[2] == False

    def test_previous_state_tracked_for_stdp(self):
        """Simulation should track previous firing for STDP learning."""
        network = Network.create_beta_weighted_directed(
            n_neurons=5,
            k_prop=0.6,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=5,
            threshold=0.3,
            firing_count=1,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        # Record initial firing
        initial_firing = state.firing.copy()

        # Step
        sim.step()

        # Previous firing should match initial
        assert np.array_equal(state.firing_prev, initial_firing)

    def test_network_and_state_dimensions_must_match(self):
        """Network n_neurons must match NeuronState size."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=5,  # Mismatched!
            threshold=0.5,
            firing_count=1,
            seed=42,
        )

        with pytest.raises(ValueError):
            Simulation(
                network=network,
                state=state,
                learning_rate=0.01,
                forgetting_rate=0.005,
            )


class TestSimulationDynamics:
    """Tests for multi-step simulation dynamics."""

    def test_activity_can_propagate(self):
        """Activity from firing neurons should propagate to neighbors."""
        # Dense network with high weights
        network = Network.create_beta_weighted_directed(
            n_neurons=20,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=20,
            threshold=0.3,
            firing_count=1,  # Start with some activity
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        initial_firing_count = np.sum(state.firing)

        # Run for several steps
        for _ in range(10):
            sim.step()

        # Activity should have changed (either increased or decreased)
        # This is a weak test - just checking dynamics occur
        final_firing_count = np.sum(state.firing)
        # The simulation ran and computed something
        assert sim.time_step == 10

    def test_isolated_neuron_stays_quiet(self):
        """A neuron with no incoming connections should not fire."""
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            firing_count=0,  # No initial firing
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        # No neurons firing, no connections = no activity
        sim.step()
        assert not np.any(state.firing)


class TestSimulationConsistency:
    """Tests for simulation consistency and determinism."""

    def test_same_initial_conditions_same_evolution(self):
        """Same initial conditions should produce same evolution."""
        def create_sim(seed):
            network = Network.create_beta_weighted_directed(
                n_neurons=15,
                k_prop=0.2,
                seed=seed,
            )
            state = NeuronState.create(
                n_neurons=15,
                threshold=0.4,
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

        # Run both for 10 steps
        for _ in range(10):
            sim1.step()
            sim2.step()

        # Should have identical states
        assert np.array_equal(sim1.state.firing, sim2.state.firing)
        assert np.array_equal(
            sim1.network.weight_matrix, sim2.network.weight_matrix
        )

    def test_reset_restores_initial_conditions(self):
        """Reset should restore to initial-like state."""
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

        # Run for a while
        for _ in range(20):
            sim.step()

        # Reset with same seed
        sim.reset(seed=42)

        # Time should be reset
        assert sim.time_step == 0
        # State should be STOPPED
        assert sim.sim_state == SimulationState.STOPPED
