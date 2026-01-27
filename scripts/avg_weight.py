#!/usr/bin/env python3
"""Parameter Sweep for forgetting rate (fixed learning rate).

Runs simulations with a fixed learning rate and varying forgetting rates, measuring:
- Average and std of weight over time (after warm-up)
- Average and std of firing neuron count
- Average and std of avalanche size
- Average and std of avalanche duration

Run with: python scripts/avg_weight.py
"""

import csv
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.weight_update import WeightUpdater


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
    config = load_config(Path("config/default.toml"))

    # Fixed parameters from config
    n_neurons = config.network.n_neurons
    weight_min = config.network.weight_min
    weight_max = config.network.weight_max
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
    )

    # Create neuron state
    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=threshold,
        firing_count=config.network.firing_count,
        seed=seed,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
    )

    # Create learner
    learner = WeightUpdater(
        enable_stdp=True,
        enable_oja=True if oja_alpha > 0 else False,
        enable_homeostatic=False,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        weight_min=weight_min,
        weight_max=weight_max,
        decay_alpha=decay_alpha,
        oja_alpha=oja_alpha,
        weight_min_inh=config.network.weight_min_inh,
        weight_max_inh=config.network.weight_max_inh,
    )

    # Create simulation
    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        learner=learner,
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
                    firing_fraction=config.network.firing_count
                    / config.network.n_neurons,
                    seed=config.seed + restart_count,
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


def _run_sweep_task(args: tuple) -> SweepResult | None:
    """Worker function for parallel execution."""
    learning_rate, forgetting_rate, total_steps, warmup_steps, idx, n_total = args
    print(
        f"[{idx}/{n_total}] lr={learning_rate:.6f} (fixed), fr={forgetting_rate:.6f}",
        flush=True,
    )
    try:
        result = run_single_sweep(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        print(
            f"  -> avg_weight={result.avg_weight:.4f}, "
            f"avg_firing={result.avg_firing_count:.1f}, "
            f"avalanches={result.n_avalanches}",
            flush=True,
        )
        return result
    except Exception as e:
        print(f"  -> ERROR: {e}", flush=True)
        return None


def main() -> None:
    """Run forgetting-rate sweep with fixed learning rate."""
    # number of steps and warm up
    total_steps = 5000
    warmup_steps = 1000
    n_replications = 3000

    # Fixed learning rate from config
    config = load_config(Path("config/default.toml"))
    fixed_lr = config.learning.learning_rate

    # 1D Latin Hypercube sampling for forgetting rate only
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=n_replications)
    fr_bounds = [0.00001, 0.1]
    forgetting_rates = (
        (fr_bounds[1] - fr_bounds[0]) * sample[:, 0] + fr_bounds[0]
    )

    # Number of parallel workers (leave some CPUs free)
    n_workers = max(1, cpu_count() - 2)

    print(f"Running {n_replications} forgetting-rate samples (lr={fixed_lr} fixed)...")
    print(f"Forgetting rate range: [{fr_bounds[0]}, {fr_bounds[1]}]")
    print(f"Total steps: {total_steps}, Warm-up: {warmup_steps}")
    print(f"Using {n_workers} parallel workers\n")

    # Prepare task arguments
    tasks = [
        (fixed_lr, fr, total_steps, warmup_steps, idx + 1, n_replications)
        for idx, fr in enumerate(forgetting_rates)
    ]

    # Run in parallel
    with Pool(processes=n_workers) as pool:
        results_raw = pool.map(_run_sweep_task, tasks)

    # Filter out failed runs
    results: list[SweepResult] = [r for r in results_raw if r is not None]

    print(f"\nCompleted {len(results)}/{n_replications} runs successfully")

    # Save results to CSV
    output_path = Path("output/forgetting_rate_sweep.csv")
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
