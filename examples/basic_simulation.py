#!/usr/bin/env python3
"""Basic simulation example - minimal headless run with metrics output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.backend import get_backend
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner


def main():
    """Run a basic simulation and print metrics."""
    # Load default configuration
    config = load_config(Path("config/default.toml"))
    backend = get_backend(prefer_gpu=False)

    # Create network and state
    network = Network.create_random(
        n_neurons=config.network.n_neurons,
        box_size=config.network.box_size,
        radius=config.network.radius,
        initial_weight=config.network.initial_weight,
        excitatory_fraction=config.network.excitatory_fraction,
        seed=config.seed,
        backend=backend,
    )

    state = NeuronState.create(
        n_neurons=config.network.n_neurons,
        threshold=config.learning.threshold,
        initial_firing_fraction=config.network.initial_firing_fraction,
        seed=config.seed,
        leak_rate=config.network.leak_rate,
        reset_potential=config.network.reset_potential,
        backend=backend,
    )

    learner = HebbianLearner(
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        weight_min=config.network.weight_min,
        weight_max=config.network.weight_max,
        decay_alpha=config.learning.decay_alpha,
        oja_alpha=config.learning.oja_alpha,
        weight_min_inh=config.network.weight_min_inh,
        weight_max_inh=config.network.weight_max_inh,
        backend=backend,
    )

    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        learner=learner,
        backend=backend,
    )

    # Run simulation
    n_steps = 1000
    print(f"Running {n_steps} steps...")

    simulation.start()
    firing_counts = []

    for step in range(n_steps):
        simulation.step()
        firing_counts.append(simulation.firing_count)

        if (step + 1) % 100 == 0:
            print(
                f"Step {step + 1}: "
                f"firing={simulation.firing_count}, "
                f"avg_weight={simulation.average_weight:.4f}"
            )

    avg_firing = sum(firing_counts) / len(firing_counts)
    print(
        f"\nFinal: {n_steps} steps, "
        f"avg firing={avg_firing:.1f}, "
        f"final avg_weight={simulation.average_weight:.4f}"
    )


if __name__ == "__main__":
    main()
