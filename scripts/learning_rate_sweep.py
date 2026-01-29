#!/usr/bin/env python3
"""2D Parameter Sweep for learning/forgetting rates using Latin Hypercube Sampling.


Run with: python scripts/learning_rate_sweep.py
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.backend import get_backend
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.hebbian import HebbianLearner


@dataclass
class SweepResult:
    """Result of a single parameter combination."""

    # Parameters
    learning_rate: float
    forgetting_rate: float

    # Metrics
    avg_weight: float
    std_weight: float
    avg_firing_count: float
    std_firing_count: float
    avg_avalanche_size: float
    std_avalanche_size: float
    avg_avalanche_duration: float
    std_avalanche_duration: float
    n_avalanches: int


def run_single_sweep(
    learning_rate: float,
    forgetting_rate: float,
    total_steps: int = 10000,
    warmup_steps: int = 1000,
    seed: int = 42,
) -> SweepResult:
    """Run a single simulation with given parameters.

    Args:
        learning_rate: Hebbian learning rate
        forgetting_rate: Weight forgetting rate
        total_steps: Total simulation steps
        warmup_steps: Steps to discard as warm-up
        seed: Random seed

    Returns:
        SweepResult with metrics
    """
    backend = get_backend(prefer_gpu=True)
    config = load_config(Path("config/default.toml"))

    # Fixed parameters from config
    n_neurons = config.network.n_neurons
    weight_min = config.network.weight_min
    weight_max = config.network.weight_max
    firing_count = config.network.firing_count
    excitatory_fraction = config.network.excitatory_fraction
    threshold = config.learning.threshold
    leak_rate = config.network.leak_rate
    reset_potential = config.network.reset_potential
    decay_alpha = config.learning.decay_alpha
    oja_alpha = config.learning.oja_alpha

    # Create network
    network = Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=config.network.k_prop,
        a=config.network.beta_a,
        b=config.network.beta_b,
        excitatory_fraction=excitatory_fraction,
        weight_min=weight_min,
        weight_max=weight_max,
        weight_min_inh=config.network.weight_min_inh,
        weight_max_inh=config.network.weight_max_inh,
        seed=seed,
        backend=backend,
    )

    # Create neuron state
    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=threshold,
        firing_count=firing_count,
        seed=seed,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
        backend=backend,
    )

    # Create learner
    learner = HebbianLearner(
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        weight_min=weight_min,
        weight_max=weight_max,
        decay_alpha=decay_alpha,
        oja_alpha=oja_alpha,
        weight_min_inh=config.network.weight_min_inh,
        weight_max_inh=config.network.weight_max_inh,
        backend=backend,
    )

    # Create simulation
    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        learner=learner,
        backend=backend,
    )

    # Create avalanche detector
    detector = AvalancheDetector(
        n_neurons=n_neurons,
        quiet_threshold=0.05,
    )

    # Data collection lists (only after warm-up)
    weight_history = []
    firing_count_history = []

    simulation.start()
    restart_count = 0

    try:
        for step in range(total_steps):
            simulation.step()
            detector.record_step(simulation.time_step, simulation.firing_count)

            # Collect data after warm-up
            if step >= warmup_steps:
                weight_history.append(simulation.average_weight)
                firing_count_history.append(simulation.firing_count)

            # Reinitialize if all neurons stop firing
            if simulation.firing_count == 0:
                restart_count += 1
                simulation.state.reinitialize_firing(
                    firing_fraction=firing_count / n_neurons,
                    seed=(config.seed or seed) + restart_count,
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Finalize avalanche detection
    detector.finalize()

    # Compute weight statistics
    weight_array = np.array(weight_history)
    avg_weight = float(np.mean(weight_array)) if len(weight_array) > 0 else 0.0
    std_weight = float(np.std(weight_array)) if len(weight_array) > 0 else 0.0

    # Compute firing count statistics
    firing_array = np.array(firing_count_history)
    avg_firing = float(np.mean(firing_array)) if len(firing_array) > 0 else 0.0
    std_firing = float(np.std(firing_array)) if len(firing_array) > 0 else 0.0

    # Filter avalanches that started after warm-up
    avalanches_after_warmup = [
        a for a in detector.avalanches if a.start_time >= warmup_steps
    ]

    # Compute avalanche size statistics
    sizes = [a.size for a in avalanches_after_warmup]
    avg_size = float(np.mean(sizes)) if len(sizes) > 0 else 0.0
    std_size = float(np.std(sizes)) if len(sizes) > 0 else 0.0

    # Compute avalanche duration statistics
    durations = [a.duration for a in avalanches_after_warmup]
    avg_duration = float(np.mean(durations)) if len(durations) > 0 else 0.0
    std_duration = float(np.std(durations)) if len(durations) > 0 else 0.0

    return SweepResult(
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        avg_weight=avg_weight,
        std_weight=std_weight,
        avg_firing_count=avg_firing,
        std_firing_count=std_firing,
        avg_avalanche_size=avg_size,
        std_avalanche_size=std_size,
        avg_avalanche_duration=avg_duration,
        std_avalanche_duration=std_duration,
        n_avalanches=len(avalanches_after_warmup),
    )


def run_sweep_worker(args: tuple) -> SweepResult | None:
    """Worker function for parallel execution.

    Args:
        args: Tuple of (index, learning_rate, forgetting_rate, total_steps, warmup_steps)

    Returns:
        SweepResult or None on error
    """
    idx, lr, fr, total_steps, warmup_steps = args
    try:
        return run_single_sweep(
            learning_rate=lr,
            forgetting_rate=fr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            seed=42 + idx,  # Different seed per run for reproducibility
        )
    except Exception as e:
        print(f"[{idx}] ERROR: {e}")
        return None


def main() -> None:
    """Run parameter sweep and save results."""
    # number of steps and warm up
    total_steps = 6000
    warmup_steps = 1000
    n_replications = 3000

    # Number of parallel workers (use all CPU cores by default)
    n_workers = os.cpu_count() or 4
    print(f"Using {n_workers} parallel workers")

    # Parameter ranges to sweep
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n_replications)
    parameters_bounds = [0, 0.1]

    # Generate all parameter combinations
    params = (parameters_bounds[1] - parameters_bounds[0]) * sample + parameters_bounds[
        0
    ]

    print(f"Running {n_replications} parameter combinations...")
    print(f"Total steps: {total_steps}, Warm-up: {warmup_steps}\n")

    # Prepare arguments for workers
    worker_args = [
        (i, lr, fr, total_steps, warmup_steps) for i, (lr, fr) in enumerate(params)
    ]

    results: list[SweepResult] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_sweep_worker, args): args for args in worker_args
        }

        # Process results as they complete
        for future in as_completed(futures):
            completed += 1
            args = futures[future]
            _, lr, fr = args[0], args[1], args[2]

            result = future.result()
            if result is not None:
                results.append(result)
                print(
                    f"[{completed}/{n_replications}] lr={lr:.4f}, fr={fr:.4f} -> "
                    f"avg_weight={result.avg_weight:.4f}, "
                    f"avg_firing={result.avg_firing_count:.1f}, "
                    f"avalanches={result.n_avalanches}"
                )
            else:
                print(
                    f"[{completed}/{n_replications}] lr={lr:.4f}, fr={fr:.4f} -> FAILED"
                )

    # Save results to CSV
    output_path = Path("output/learning_forgetting_sweep.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "learning_rate",
                "forgetting_rate",
                "avg_weight",
                "std_weight",
                "avg_firing_count",
                "std_firing_count",
                "avg_avalanche_size",
                "std_avalanche_size",
                "avg_avalanche_duration",
                "std_avalanche_duration",
                "n_avalanches",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.learning_rate,
                    r.forgetting_rate,
                    f"{r.avg_weight:.6f}",
                    f"{r.std_weight:.6f}",
                    f"{r.avg_firing_count:.4f}",
                    f"{r.std_firing_count:.4f}",
                    f"{r.avg_avalanche_size:.4f}",
                    f"{r.std_avalanche_size:.4f}",
                    f"{r.avg_avalanche_duration:.4f}",
                    f"{r.std_avalanche_duration:.4f}",
                    r.n_avalanches,
                ]
            )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
