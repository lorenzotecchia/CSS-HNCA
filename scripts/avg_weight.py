#!/usr/bin/env python3
"""Parameter Sweep for learning/forgetting rates with replications.

Runs simulations with different learning/forgetting rate combinations,
performs multiple replications for each, and computes averages across replications.

Run with: python scripts/avg_weight.py
"""

import csv
import sys
from dataclasses import dataclass, field
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.stats import qmc

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.weight_update import WeightUpdater


@dataclass
class SingleRunResult:
    """Result of a single simulation run."""

    learning_rate: float
    forgetting_rate: float
    avg_weight: float
    std_weight: float
    avg_firing_count: float
    std_firing_count: float
    avg_avalanche_size: float
    std_avalanche_size: float
    avg_avalanche_duration: float
    std_avalanche_duration: float
    n_avalanches: int


@dataclass
class AggregatedResult:
    """Aggregated result across replications."""

    learning_rate: float
    forgetting_rate: float
    n_replications: int

    # Means across replications
    mean_avg_weight: float
    mean_std_weight: float
    mean_avg_firing_count: float
    mean_std_firing_count: float
    mean_avg_avalanche_size: float
    mean_std_avalanche_size: float
    mean_avg_avalanche_duration: float
    mean_std_avalanche_duration: float
    mean_n_avalanches: float

    # Std across replications
    std_avg_weight: float
    std_avg_firing_count: float
    std_avg_avalanche_size: float
    std_avg_avalanche_duration: float
    std_n_avalanches: float


def run_single_sweep(
    learning_rate: float,
    forgetting_rate: float,
    total_steps: int = 10000,
    warmup_steps: int = 1000,
    seed: int = 42,
) -> SingleRunResult:
    """Run a single simulation with given parameters."""
    config = load_config(Path("config/default.toml"))

    n_neurons = config.network.n_neurons
    weight_min = config.network.weight_min
    weight_max = config.network.weight_max
    excitatory_fraction = config.network.excitatory_fraction
    threshold = config.learning.threshold
    leak_rate = config.network.leak_rate
    reset_potential = config.network.reset_potential
    decay_alpha = config.learning.decay_alpha
    oja_alpha = config.learning.oja_alpha

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

    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=threshold,
        firing_count=config.network.firing_count,
        seed=seed,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
    )

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

    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        learner=learner,
    )

    detector = AvalancheDetector(
        n_neurons=n_neurons,
        quiet_threshold=0.05,
    )

    weight_history = []
    firing_count_history = []

    simulation.start()
    restart_count = 0

    for step in range(total_steps):
        simulation.step()
        detector.record_step(simulation.time_step, simulation.firing_count)

        if step >= warmup_steps:
            weight_history.append(simulation.average_weight)
            firing_count_history.append(simulation.firing_count)

        if simulation.firing_count == 0:
            restart_count += 1
            simulation.state.reinitialize_firing(
                firing_fraction=config.network.firing_count / config.network.n_neurons,
                seed=config.seed + restart_count,
            )

    detector.finalize()

    weight_array = np.array(weight_history)
    avg_weight = float(np.mean(weight_array)) if len(weight_array) > 0 else 0.0
    std_weight = float(np.std(weight_array)) if len(weight_array) > 0 else 0.0

    firing_array = np.array(firing_count_history)
    avg_firing = float(np.mean(firing_array)) if len(firing_array) > 0 else 0.0
    std_firing = float(np.std(firing_array)) if len(firing_array) > 0 else 0.0

    avalanches_after_warmup = [
        a for a in detector.avalanches if a.start_time >= warmup_steps
    ]

    sizes = [a.size for a in avalanches_after_warmup]
    avg_size = float(np.mean(sizes)) if len(sizes) > 0 else 0.0
    std_size = float(np.std(sizes)) if len(sizes) > 0 else 0.0

    durations = [a.duration for a in avalanches_after_warmup]
    avg_duration = float(np.mean(durations)) if len(durations) > 0 else 0.0
    std_duration = float(np.std(durations)) if len(durations) > 0 else 0.0

    return SingleRunResult(
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


def _run_replication_task(args: tuple) -> SingleRunResult | None:
    """Worker function for a single replication."""
    learning_rate, forgetting_rate, total_steps, warmup_steps, seed = args
    try:
        return run_single_sweep(
            learning_rate=learning_rate,
            forgetting_rate=forgetting_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            seed=seed,
        )
    except Exception as e:
        print(f"ERROR (lr={learning_rate}, fr={forgetting_rate}, seed={seed}): {e}")
        return None


def aggregate_results(results: list[SingleRunResult]) -> AggregatedResult:
    """Aggregate results from multiple replications."""
    lr = results[0].learning_rate
    fr = results[0].forgetting_rate

    avg_weights = [r.avg_weight for r in results]
    std_weights = [r.std_weight for r in results]
    avg_firings = [r.avg_firing_count for r in results]
    std_firings = [r.std_firing_count for r in results]
    avg_sizes = [r.avg_avalanche_size for r in results]
    std_sizes = [r.std_avalanche_size for r in results]
    avg_durations = [r.avg_avalanche_duration for r in results]
    std_durations = [r.std_avalanche_duration for r in results]
    n_avalanches = [r.n_avalanches for r in results]

    return AggregatedResult(
        learning_rate=lr,
        forgetting_rate=fr,
        n_replications=len(results),
        mean_avg_weight=float(np.mean(avg_weights)),
        mean_std_weight=float(np.mean(std_weights)),
        mean_avg_firing_count=float(np.mean(avg_firings)),
        mean_std_firing_count=float(np.mean(std_firings)),
        mean_avg_avalanche_size=float(np.mean(avg_sizes)),
        mean_std_avalanche_size=float(np.mean(std_sizes)),
        mean_avg_avalanche_duration=float(np.mean(avg_durations)),
        mean_std_avalanche_duration=float(np.mean(std_durations)),
        mean_n_avalanches=float(np.mean(n_avalanches)),
        std_avg_weight=float(np.std(avg_weights)),
        std_avg_firing_count=float(np.std(avg_firings)),
        std_avg_avalanche_size=float(np.std(avg_sizes)),
        std_avg_avalanche_duration=float(np.std(avg_durations)),
        std_n_avalanches=float(np.std(n_avalanches)),
    )


def main() -> None:
    """Run parameter sweep with replications."""
    # Configuration
    total_steps = 5000
    warmup_steps = 1000
    n_replications = 40  # Number of replications per parameter combination
    n_sweeps = 100

    # Parameter ranges to sweep
    n_total_runs = n_sweeps * n_replications
    n_workers = max(1, cpu_count() - 2)

    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n_sweeps)
    parameters_bounds = [0.00001, 0.1]
    rates = (parameters_bounds[1] - parameters_bounds[0]) * sample + parameters_bounds[
        0
    ]

    print(
        f"Running {n_sweeps} parameter combinations x {n_replications} replications = {n_total_runs} total runs"
    )
    print(f"Total steps: {total_steps}, Warm-up: {warmup_steps}")
    print(f"Using {n_workers} parallel workers\n")

    # Generator to yield tasks lazily instead of building full list in memory
    def task_generator():
        for lr, fr in rates:
            for rep in range(n_replications):
                seed = 42 + rep * 1000  # Different seed for each replication
                yield (lr, fr, total_steps, warmup_steps, seed)

    # Run tasks in parallel and aggregate results incrementally
    results_by_params: dict[tuple[float, float], list[SingleRunResult]] = {}
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_run_replication_task, task_generator(), chunksize=10):
            if result is not None:
                key = (result.learning_rate, result.forgetting_rate)
                if key not in results_by_params:
                    results_by_params[key] = []
                results_by_params[key].append(result)
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{n_total_runs} tasks completed")

    # Aggregate each parameter combination
    aggregated_results: list[AggregatedResult] = []
    for (lr, fr), results in sorted(results_by_params.items()):
        if len(results) > 0:
            agg = aggregate_results(results)
            aggregated_results.append(agg)
            print(
                f"lr={lr:.4f}, fr={fr:.4f}: "
                f"avg_weight={agg.mean_avg_weight:.4f}±{agg.std_avg_weight:.4f}, "
                f"replications={agg.n_replications}"
            )

    # Save aggregated results to CSV
    output_path = Path("output/learning_forgetting_sweep_aggregated.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "learning_rate",
                "forgetting_rate",
                "n_replications",
                "mean_avg_weight",
                "std_avg_weight",
                "mean_std_weight",
                "mean_avg_firing_count",
                "std_avg_firing_count",
                "mean_std_firing_count",
                "mean_avg_avalanche_size",
                "std_avg_avalanche_size",
                "mean_std_avalanche_size",
                "mean_avg_avalanche_duration",
                "std_avg_avalanche_duration",
                "mean_std_avalanche_duration",
                "mean_n_avalanches",
                "std_n_avalanches",
            ]
        )
        for r in aggregated_results:
            writer.writerow(
                [
                    r.learning_rate,
                    r.forgetting_rate,
                    r.n_replications,
                    f"{r.mean_avg_weight:.6f}",
                    f"{r.std_avg_weight:.6f}",
                    f"{r.mean_std_weight:.6f}",
                    f"{r.mean_avg_firing_count:.4f}",
                    f"{r.std_avg_firing_count:.4f}",
                    f"{r.mean_std_firing_count:.4f}",
                    f"{r.mean_avg_avalanche_size:.4f}",
                    f"{r.std_avg_avalanche_size:.4f}",
                    f"{r.mean_std_avalanche_size:.4f}",
                    f"{r.mean_avg_avalanche_duration:.4f}",
                    f"{r.std_avg_avalanche_duration:.4f}",
                    f"{r.mean_std_avalanche_duration:.4f}",
                    f"{r.mean_n_avalanches:.4f}",
                    f"{r.std_n_avalanches:.4f}",
                ]
            )

    print(f"\nAggregated results saved to: {output_path}")


if __name__ == "__main__":
    main()
