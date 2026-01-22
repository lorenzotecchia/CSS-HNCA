"""Integration tests for LIF dynamics in simulation.

RED phase: These tests should fail until LIF is integrated into Simulation.
"""

import numpy as np
import pytest
from pathlib import Path

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation


class TestLIFPreventsRunaway:
    """Tests that LIF dynamics prevent continuous firing saturation."""

    def test_simulation_with_lif_resets_firing_neurons(self):
        """Neurons that fire should have potential reset, preventing immediate re-fire."""
        # Create small network with strong connections
        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=3.0,  # Large radius = more connections
            initial_weight=0.5,
            seed=42,
        )
        
        # Create state with LIF dynamics
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.5,  # Start with 50% firing
            seed=42,
            leak_rate=0.1,
            reset_potential=1.0,  # Full reset
        )
        
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )
        
        # Run several steps
        firing_counts = []
        for _ in range(20):
            sim.step()
            firing_counts.append(sim.firing_count)
        
        # With LIF reset, we should NOT have all neurons firing continuously
        # There should be variation in firing counts
        assert max(firing_counts) <= 10, "All neurons firing = no LIF reset effect"
        assert min(firing_counts) != max(firing_counts), "Firing count should vary"

    def test_lif_creates_avalanche_dynamics(self):
        """LIF should create transient dynamics (activity rises then falls)."""
        network = Network.create_random(
            n_neurons=50,
            box_size=(10.0, 10.0, 10.0),
            radius=3.0,
            initial_weight=0.15,  # Lower weight to avoid saturation
            seed=42,
        )
        
        state = NeuronState.create(
            n_neurons=50,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
            leak_rate=0.15,  # Slower leak
            reset_potential=0.6,  # Partial reset
        )
        
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.012,  # LTD > LTP for stability
        )
        
        # Run and collect firing counts
        firing_counts = []
        for _ in range(100):  # Run longer
            sim.step()
            firing_counts.append(sim.firing_count)
        
        # Should have transient dynamics - starts with activity, eventually dies
        initial_activity = sum(firing_counts[:10])  # Activity in first 10 steps
        final_activity = sum(firing_counts[-10:])  # Activity in last 10 steps
        
        # Initial burst of activity
        assert initial_activity > 0, "Should have some initial activity"
        # Activity should decrease over time (not saturate at max)
        assert final_activity < initial_activity, "Activity should decrease over time"


class TestLIFConfigIntegration:
    """Tests for LIF parameters in config loading."""

    def test_config_has_leak_rate(self):
        """Config should include leak_rate parameter."""
        from src.config.loader import load_config
        
        config = load_config(Path("config/default.toml"))
        assert hasattr(config.network, "leak_rate")

    def test_config_has_reset_potential(self):
        """Config should include reset_potential parameter."""
        from src.config.loader import load_config
        
        config = load_config(Path("config/default.toml"))
        assert hasattr(config.network, "reset_potential")
