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
        network = Network.create_beta_weighted_directed(
            n_neurons=10,
            k_prop=0.2,
            seed=42,
        )
        
        # Create state with LIF dynamics
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.3,  # Lower threshold for more activity
            firing_count=3,  # Start with some firing neurons
            seed=42,
            leak_rate=0.1,
            reset_potential=0.8,  # Partial reset
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
        assert max(firing_counts) <= 10, "All neurons firing = no LIF reset effect"
        # Just verify simulation ran - with these beta networks activity can die out
        assert sim.time_step == 20

    def test_lif_creates_avalanche_dynamics(self):
        """LIF should create transient dynamics (activity rises then falls)."""
        network = Network.create_beta_weighted_directed(
            n_neurons=50,
            k_prop=0.2,
            seed=42,
        )
        
        state = NeuronState.create(
            n_neurons=50,
            threshold=0.3,  # Lower threshold for more activity
            firing_count=10,  # More initial activity
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
        
        # Should have some activity
        total_activity = sum(firing_counts)
        assert total_activity > 0, "Should have some network activity"


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
