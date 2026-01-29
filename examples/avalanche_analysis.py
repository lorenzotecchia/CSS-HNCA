#!/usr/bin/env python3
"""Demo script for Avalanche Analytics visualization.

Run with: python scripts/demo_avalanche_analytics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.hebbian import HebbianLearner
from src.visualization.avalanche_view import AvalancheAnalyticsView


def main() -> None:
    """Run avalanche analytics demo."""
    # Load config
    config = load_config(Path("config/default.toml"))

    # Create simulation
    network = Network.create_random(
        n_neurons=config.network.n_neurons,
        box_size=config.network.box_size,
        radius=config.network.radius,
        initial_weight=config.network.initial_weight,
        excitatory_fraction=config.network.excitatory_fraction,
        seed=config.seed,
    )
    learner = HebbianLearner(
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        weight_min=config.network.weight_min,
        weight_max=config.network.weight_max,
        weight_min_inh=config.network.weight_min_inh,
        weight_max_inh=config.network.weight_max_inh,
        decay_alpha=config.learning.decay_alpha,
        oja_alpha=config.learning.oja_alpha,
    )

    state = NeuronState.create(
        n_neurons=config.network.n_neurons,
        threshold=config.learning.threshold,
        initial_firing_fraction=config.network.initial_firing_fraction,
        seed=config.seed,
        leak_rate=config.network.leak_rate,
        reset_potential=config.network.reset_potential,
    )

    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        learner=learner,
    )

    # Create avalanche detector
    detector = AvalancheDetector(
        n_neurons=config.network.n_neurons,
        quiet_threshold=0.05,
    )

    # Create visualization
    view = AvalancheAnalyticsView(
        detector=detector,
        target_avalanches=100,
        update_interval=10,
    )
    view.initialize()

    print("Running avalanche analytics...")
    print(f"Target: {view.target_avalanches} avalanches")
    print("Close the plot window to exit early.")

    restart_count = 0
    try:
        while not view.should_stop():
            simulation.step()
            detector.record_step(simulation.time_step, simulation.firing_count)
            view.update(
                time_step=simulation.time_step,
                firing_count=simulation.firing_count,
                n_neurons=simulation.network.n_neurons,
                avg_weight=simulation.average_weight,
            )

            # When all neurons stop firing, reinitialize with random fraction
            if simulation.firing_count == 0:
                restart_count += 1
                if restart_count >= view.target_avalanches:
                    break
                simulation.state.reinitialize_firing(
                    firing_fraction=config.network.initial_firing_fraction,
                    seed=config.seed + restart_count,
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Finalize and save
    detector.finalize()
    output_path = Path("output/avalanches.csv")
    view.save_csv(output_path)
    print(
        f"\nCollected {len(detector.avalanches)} avalanches in {simulation.time_step} steps"
    )
    print(f"Saved to: {output_path}")

    view.show()


if __name__ == "__main__":
    main()
