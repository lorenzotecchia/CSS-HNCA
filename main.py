"""Neural Cellular Automata CLI.

Command-line interface for running neural cellular automata simulations.
"""

import argparse
import sys
from pathlib import Path

from src.config.loader import load_config, ConfigValidationError
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner


DEFAULT_CONFIG_PATH = Path("config/default.toml")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="css-hnca",
        description="Neural Cellular Automata simulation with Hebbian learning and STDP",
        epilog="Example: python main.py -c config/custom.toml --headless --steps 1000",
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML configuration file (default: config/default.toml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without visualization",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps to run (required for headless mode)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (TUI logging)",
    )

    return parser


def run_headless(simulation: Simulation, steps: int, verbose: bool) -> None:
    """Run simulation in headless mode.

    Args:
        simulation: The simulation to run
        steps: Number of steps to execute
        verbose: Whether to print step information
    """
    simulation.start()

    for _ in range(steps):
        simulation.step()

        if verbose:
            print(
                f"[t={simulation.time_step:05d}] "
                f"firing: {simulation.firing_count} | "
                f"avg_weight: {simulation.average_weight:.4f}"
            )

    if not verbose:
        print(f"Simulation completed: {steps} steps")
        print(f"Final firing count: {simulation.firing_count}")
        print(f"Final average weight: {simulation.average_weight:.4f}")


def create_simulation_from_config(config_path: Path) -> Simulation:
    """Create a Simulation from a configuration file.

    Args:
        config_path: Path to the TOML configuration file

    Returns:
        Configured Simulation instance
    """
    config = load_config(config_path)

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

    return Simulation(
        network=network,
        state=state,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        learner=learner,
    )


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.headless and args.steps is None:
        parser.error("--steps is required when running in --headless mode")

    # Load configuration
    try:
        simulation = create_simulation_from_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ConfigValidationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded configuration from: {args.config}")
        print(f"Network: {simulation.network.n_neurons} neurons")
        print(f"Box size: {simulation.network.box_size}")
        print(f"Connectivity radius: {simulation.network.radius}")

    # Run simulation
    if args.headless:
        run_headless(simulation, args.steps, args.verbose)
    else:
        # Full visualization mode - to be implemented in Phase 4
        print("Visualization mode not yet implemented.")
        print("Use --headless --steps N to run in headless mode.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
