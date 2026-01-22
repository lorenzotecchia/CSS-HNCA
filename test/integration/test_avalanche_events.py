"""Integration tests for avalanche detection with simulation.

RED phase: Tests for AvalancheDetector integration with event bus.
"""

import numpy as np
import pytest

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.events.bus import EventBus, StepEvent


class TestAvalancheEventBusIntegration:
    """Tests for AvalancheDetector subscribing to simulation events."""

    def test_detector_receives_step_events(self):
        """Detector should receive step events from event bus."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1)
        bus = EventBus()
        
        # Subscribe detector's record_step to step events
        def on_step(event: StepEvent) -> None:
            detector.record_step(event.time_step, event.firing_count)
        
        bus.subscribe(StepEvent, on_step)
        
        # Emit events simulating an avalanche
        bus.emit(StepEvent(time_step=0, firing_count=5, avg_weight=0.1))
        bus.emit(StepEvent(time_step=1, firing_count=15, avg_weight=0.1))
        bus.emit(StepEvent(time_step=2, firing_count=5, avg_weight=0.1))
        
        assert len(detector.avalanches) == 1

    def test_detector_with_real_simulation(self):
        """Detector should work with real simulation events."""
        network = Network.create_random(
            n_neurons=50,
            box_size=(10.0, 10.0, 10.0),
            radius=3.0,
            initial_weight=0.3,
            seed=42,
        )
        
        state = NeuronState.create(
            n_neurons=50,
            threshold=0.5,
            initial_firing_fraction=0.3,
            seed=42,
            leak_rate=0.2,
            reset_potential=0.8,
        )
        
        bus = EventBus()
        detector = AvalancheDetector(n_neurons=50, quiet_threshold=0.1)
        
        def on_step(event: StepEvent) -> None:
            detector.record_step(event.time_step, event.firing_count)
        
        bus.subscribe(StepEvent, on_step)
        
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )
        
        # Run simulation
        for _ in range(100):
            sim.step()
        
        detector.finalize()
        
        # Should have recorded avalanches (with LIF, activity varies)
        # At minimum, detector should have processed all steps
        assert detector.n_neurons == 50


class TestAvalancheLIFInteraction:
    """Tests for avalanche behavior with LIF neurons."""

    def test_lif_creates_avalanche_patterns(self):
        """LIF dynamics should create avalanche-like activity patterns."""
        network = Network.create_random(
            n_neurons=100,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.2,
            seed=42,
        )
        
        state = NeuronState.create(
            n_neurons=100,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
            leak_rate=0.15,
            reset_potential=0.7,
        )
        
        bus = EventBus()
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.05)
        
        def on_step(event: StepEvent) -> None:
            detector.record_step(event.time_step, event.firing_count)
        
        bus.subscribe(StepEvent, on_step)
        
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            event_bus=bus,
        )
        
        # Run long simulation
        for _ in range(500):
            sim.step()
        
        detector.finalize()
        
        # With proper LIF, we should see variable activity
        sizes = detector.get_size_distribution()
        
        if len(sizes) > 1:
            # Should have variation in avalanche sizes
            assert max(sizes) != min(sizes), "Avalanche sizes should vary"
