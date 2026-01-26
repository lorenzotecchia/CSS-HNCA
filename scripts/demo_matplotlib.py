#!/usr/bin/env python3
"""Demo script for Matplotlib Analytics visualization.

Run with: python scripts/demo_matplotlib.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner
from src.visualization.matplotlib_view import MatplotlibAnalyticsView


def main() -> None:
    """Run demo visualization."""
    # Load config
    config = load_config(Path("config/default.toml"))

    # Create simulation
    network = Network.create_random(
        n_neurons=config.network.n_neurons,
        box_size=config.network.box_size,
        radius=config.network.radius,
        initial_weight=config.network.initial_weight,
        seed=config.seed,
    )
    state = NeuronState.create(
        n_neurons=config.network.n_neurons,
        threshold=config.learning.threshold,
        initial_firing_fraction=config.network.initial_firing_fraction,
        seed=config.seed,
        leak_rate=config.network.leak_rate,
        reset_potential=config.network.reset_potential,
    )
    learner = HebbianLearner(
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        weight_min=config.network.weight_min,
        weight_max=config.network.weight_max,
        decay_alpha=config.learning.decay_alpha,
        oja_alpha=config.learning.oja_alpha,
    )
    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        learner=learner,
    )

    # Create visualization
    view = MatplotlibAnalyticsView(show_heatmap=True)
    view.initialize()

    print("Running simulation with matplotlib visualization...")
    print("Close the plot window to exit.")

    try:
        for _ in range(500):
            simulation.step()
            view.update_from_simulation(simulation)
            plt.pause(0.05)  # Non-blocking: processes GUI events
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        view.show()  # Keep window open at end


if __name__ == "__main__":
    main()
