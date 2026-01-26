"""Integration tests for Simulation event emission.

Tests that Simulation emits events on step and reset.
"""

import pytest

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.bus import EventBus, ResetEvent, StepEvent


class TestSimulationEmitsStepEvents:
    """Test that Simulation emits StepEvent on each step."""

    def test_step_emits_step_event(self) -> None:
        """Each step should emit a StepEvent with correct data."""
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
        bus = EventBus()
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )

        received: list[StepEvent] = []

        def handler(event: StepEvent) -> None:
            received.append(event)

        bus.subscribe(StepEvent, handler)

        sim.step()

        assert len(received) == 1
        assert received[0].time_step == 1
        assert received[0].firing_count == sim.firing_count
        assert received[0].avg_weight == pytest.approx(sim.average_weight)

    def test_multiple_steps_emit_multiple_events(self) -> None:
        """Multiple steps should emit corresponding StepEvents."""
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
        bus = EventBus()
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )

        received: list[StepEvent] = []
        bus.subscribe(StepEvent, lambda e: received.append(e))

        for _ in range(5):
            sim.step()

        assert len(received) == 5
        assert [e.time_step for e in received] == [1, 2, 3, 4, 5]


class TestSimulationEmitsResetEvents:
    """Test that Simulation emits ResetEvent on reset."""

    def test_reset_emits_reset_event(self) -> None:
        """Reset should emit a ResetEvent with seed."""
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
        bus = EventBus()
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )

        received: list[ResetEvent] = []
        bus.subscribe(ResetEvent, lambda e: received.append(e))

        sim.reset(seed=123)

        assert len(received) == 1
        assert received[0].seed == 123

    def test_reset_without_seed_emits_event_with_none(self) -> None:
        """Reset without seed should emit ResetEvent with seed=None."""
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
        bus = EventBus()
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )

        received: list[ResetEvent] = []
        bus.subscribe(ResetEvent, lambda e: received.append(e))

        sim.reset()

        assert len(received) == 1
        assert received[0].seed is None


class TestSimulationWithoutEventBus:
    """Test that Simulation works without an event bus."""

    def test_step_works_without_event_bus(self) -> None:
        """Simulation should work normally if no event bus provided."""
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

        # Should not raise
        sim.step()
        sim.reset(seed=99)

        assert sim.time_step == 0
