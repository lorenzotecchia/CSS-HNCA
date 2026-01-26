"""Property-based tests for Simulation using Hypothesis.

RED phase: These tests should fail until Simulation is implemented.
"""

import numpy as np
from hypothesis import given, strategies as st, settings

from src.core.simulation import Simulation, SimulationState
from src.core.network import Network
from src.core.neuron_state import NeuronState


# Custom strategies
n_neurons_strategy = st.integers(min_value=5, max_value=50)
steps_strategy = st.integers(min_value=1, max_value=100)
threshold_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
rate_strategy = st.floats(min_value=0.001, max_value=0.1, allow_nan=False)
firing_count_strategy = st.integers(min_value=1, max_value=50)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


def create_simulation(
    n_neurons: int,
    threshold: float,
    learning_rate: float,
    forgetting_rate: float,
    firing_count: int,
    seed: int,
    excitatory_fraction: float = 1.0,
) -> Simulation:
    """Helper to create a simulation with given parameters."""
    # Compute valid k_prop for given n_neurons (must be in [2/n, 1-1/n])
    k_min = 2 / n_neurons
    k_max = 1 - 1 / n_neurons
    k_prop = (k_min + k_max) / 2  # Use midpoint of valid range
    # Ensure firing_count doesn't exceed n_neurons
    firing_count = min(firing_count, n_neurons)
    network = Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=k_prop,
        a=2.0,
        b=6.0,
        seed=seed,
        excitatory_fraction=excitatory_fraction,
    )
    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=threshold,
        firing_count=firing_count,
        seed=seed,
    )
    return Simulation(
        network=network,
        state=state,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
    )


class TestSimulationStepProperties:
    """Property tests for step behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_step_count_always_increases(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Time step must monotonically increase with each step()."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for expected_step in range(1, n_steps + 1):
            sim.step()
            assert sim.time_step == expected_step

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_firing_state_shape_preserved(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Firing state shape must remain (n_neurons,) after any steps."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim.step()

        assert sim.state.firing.shape == (n_neurons,)
        assert sim.state.firing_prev.shape == (n_neurons,)

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_weight_matrix_shape_preserved(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Weight matrix shape must remain (n_neurons, n_neurons) after steps."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim.step()

        assert sim.network.weight_matrix.shape == (n_neurons, n_neurons)


class TestSimulationResetProperties:
    """Property tests for reset behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_reset_always_zeroes_time(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Reset must always set time_step to 0."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim.step()

        sim.reset()
        assert sim.time_step == 0

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_reset_always_stops_simulation(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Reset must always set state to STOPPED."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        sim.start()
        for _ in range(n_steps):
            sim.step()

        sim.reset()
        assert sim.sim_state == SimulationState.STOPPED


class TestSimulationStateTransitionProperties:
    """Property tests for state machine behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_start_always_leads_to_running(
        self,
        n_neurons,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """start() must always result in RUNNING state."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        sim.start()
        assert sim.sim_state == SimulationState.RUNNING

    @given(
        n_neurons=n_neurons_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_pause_from_running_leads_to_paused(
        self,
        n_neurons,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """pause() from RUNNING must result in PAUSED state."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        sim.start()
        sim.pause()
        assert sim.sim_state == SimulationState.PAUSED


class TestSimulationDeterminismProperties:
    """Property tests for deterministic behavior."""

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=st.integers(min_value=1, max_value=20),
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=20)
    def test_same_seed_same_evolution(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Same seed must produce identical simulation evolution."""
        sim1 = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )
        sim2 = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim1.step()
            sim2.step()

        assert np.array_equal(sim1.state.firing, sim2.state.firing)
        assert np.array_equal(sim1.state.firing_prev, sim2.state.firing_prev)
        assert np.array_equal(
            sim1.network.weight_matrix, sim2.network.weight_matrix
        )


class TestSimulationMetricProperties:
    """Property tests for simulation metrics."""

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_firing_count_within_bounds(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Firing count must always be between 0 and n_neurons."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim.step()
            assert 0 <= sim.firing_count <= n_neurons

    @given(
        n_neurons=n_neurons_strategy,
        n_steps=steps_strategy,
        threshold=threshold_strategy,
        learning_rate=rate_strategy,
        forgetting_rate=rate_strategy,
        firing_count=firing_count_strategy,
        seed=seed_strategy,
    )
    @settings(deadline=None, max_examples=30)
    def test_average_weight_non_negative(
        self,
        n_neurons,
        n_steps,
        threshold,
        learning_rate,
        forgetting_rate,
        firing_count,
        seed,
    ):
        """Average weight must always be non-negative."""
        sim = create_simulation(
            n_neurons, threshold, learning_rate, forgetting_rate,
            firing_count, seed
        )

        for _ in range(n_steps):
            sim.step()
            assert sim.average_weight >= 0
