#!/usr/bin/env python3
"""SOC Parameter Sweep for finding critical dynamics.

Runs simulations with different parameter combinations and measures:
- Avalanche size distribution slope (target: -1.5 for power law)
- Branching ratio (target: ~1.0 for criticality)
- Average activity stability (homeostasis)

Run with: python scripts/soc_parameter_sweep.py
"""

import csv
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.weight_update import WeightUpdater
from src.visualization.avalanche_view import AvalancheAnalyticsView


@dataclass
class SweepResult:
    """Result of a single parameter combination."""

    # Parameters
    leak_rate: float
    reset_potential: float
    decay_alpha: float
    oja_alpha: float
    threshold: float

    # Metrics
    n_avalanches: int
    branching_ratio: float
    power_law_slope: float
    avg_activity: float
    activity_std: float
    final_avg_weight: float


def compute_power_law_slope(sizes: list[int]) -> float:
    """Estimate power-law slope from avalanche size distribution.

    Uses linear regression on log-log scale.
    For true SOC, slope should be around -1.5.

    Args:
        sizes: List of avalanche sizes

    Returns:
        Estimated slope, or 0.0 if insufficient data
    """
    if len(sizes) < 5:
        return 0.0

    # Filter out zero/negative sizes
    sizes = [s for s in sizes if s > 0]
    if len(sizes) < 5:
        return 0.0

    # Create histogram bins
    n_bins = min(20, len(set(sizes)))
    if n_bins < 3:
        return 0.0

    hist, bin_edges = np.histogram(sizes, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter out zero counts
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0

    log_x = np.log(bin_centers[mask])
    log_y = np.log(hist[mask])

    # Simple linear regression
    n = len(log_x)
    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xy = np.sum(log_x * log_y)
    sum_x2 = np.sum(log_x**2)

    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-10:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


def run_single_sweep(
    leak_rate: float,
    reset_potential: float,
    decay_alpha: float,
    oja_alpha: float,
    threshold: float,
    view: AvalancheAnalyticsView,
    target_avalanches: int = 100,
    seed: int = 42,
) -> SweepResult:
    """Run a single simulation with given parameters.

    Args:
        leak_rate: LIF leak rate
        reset_potential: LIF reset potential
        decay_alpha: Baseline weight decay
        oja_alpha: Oja normalization strength
        threshold: Firing threshold
        view: Shared visualization view to reuse
        target_avalanches: Number of avalanches to simulate,
        seed: Random seed

    Returns:
        SweepResult with metrics
    """
    config = load_config(Path("config/default.toml"))

    # Fixed parameters (from default config)
    n_neurons = config.network.n_neurons
    weight_min = config.network.weight_min
    weight_max = config.network.weight_max
    learning_rate = config.learning.learning_rate
    forgetting_rate = config.learning.forgetting_rate

    # Create network
    network = Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=config.network.k_prop,
        a=config.network.beta_a,
        b=config.network.beta_b,
        excitatory_fraction=config.network.excitatory_fraction,
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
        quiet_threshold=0.05,  # 5% of neurons = "quiet"
    )

    # Reset the shared view with new detector
    view.detector = detector
    view.target_avalanches = target_avalanches
    view._time_steps = []
    view._firing_counts = []
    view._nonfiring_counts = []
    view._avg_weights = []
    view._last_update_step = 0

    # Run simulation and collect metrics
    activity_history = []
    simulation.start()

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
                simulation.state.reinitialize_firing(
                    firing_fraction=config.network.firing_count / config.network.n_neurons,
                    seed=config.seed + restart_count,
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Finalize avalanche detection
    detector.finalize()

    # Compute metrics
    sizes = detector.get_size_distribution()
    branching_ratio = detector.compute_branching_ratio()
    power_law_slope = compute_power_law_slope(sizes)

    activity_array = np.array(activity_history)
    avg_activity = float(np.mean(activity_array))
    activity_std = float(np.std(activity_array))

    return SweepResult(
        leak_rate=leak_rate,
        reset_potential=reset_potential,
        decay_alpha=decay_alpha,
        oja_alpha=oja_alpha,
        threshold=threshold,
        n_avalanches=len(detector.avalanches),
        branching_ratio=branching_ratio,
        power_law_slope=power_law_slope,
        avg_activity=avg_activity,
        activity_std=activity_std,
        final_avg_weight=simulation.average_weight,
    )


def main() -> None:
    """Run parameter sweep and save results."""
    # Parameter ranges to sweep
    leak_rates = [0.05, 0.08, 0.12, 0.15]
    reset_potentials = [0.3, 0.4, 0.5, 0.6]
    decay_alphas = [0.0002, 0.0005, 0.001]
    oja_alphas = [0.001, 0.002, 0.004]
    thresholds = [0.35, 0.4, 0.45]

    # Calculate total combinations
    total = (
        len(leak_rates)
        * len(reset_potentials)
        * len(decay_alphas)
        * len(oja_alphas)
        * len(thresholds)
    )
    print(f"Running {total} parameter combinations...")

    results: list[SweepResult] = []
    all_avalanches: list[tuple[float, float, float, float, float, int, int]] = []
    count = 0

    # Create a single shared view
    dummy_detector = AvalancheDetector(n_neurons=350, quiet_threshold=0.05)
    view = AvalancheAnalyticsView(
        detector=dummy_detector,
        target_avalanches=100,
        update_interval=10,
    )
    view.initialize()

    for leak, reset, decay, oja, thresh in product(
        leak_rates, reset_potentials, decay_alphas, oja_alphas, thresholds
    ):
        count += 1
        print(
            f"[{count}/{total}] leak={leak:.2f}, reset={reset:.1f}, "
            f"decay={decay:.4f}, oja={oja:.3f}, thresh={thresh:.2f}",
            end=" -> ",
        )

        try:
            result = run_single_sweep(
                leak_rate=leak,
                reset_potential=reset,
                decay_alpha=decay,
                oja_alpha=oja,
                threshold=thresh,
                view=view,
            )
            results.append(result)

            # Collect avalanche data with parameters
            for a in view.detector.avalanches:
                all_avalanches.append(
                    (leak, reset, decay, oja, thresh, a.size, a.duration)
                )

            print(
                f"avalanches={result.n_avalanches}, "
                f"BR={result.branching_ratio:.3f}, "
                f"slope={result.power_law_slope:.2f}"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    # Close the shared view
    view.close()

    # Save results to CSV
    output_path = Path("output/soc_sweep_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "leak_rate",
                "reset_potential",
                "decay_alpha",
                "oja_alpha",
                "threshold",
                "n_avalanches",
                "branching_ratio",
                "power_law_slope",
                "avg_activity",
                "activity_std",
                "final_avg_weight",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.leak_rate,
                    r.reset_potential,
                    r.decay_alpha,
                    r.oja_alpha,
                    r.threshold,
                    r.n_avalanches,
                    f"{r.branching_ratio:.4f}",
                    f"{r.power_law_slope:.4f}",
                    f"{r.avg_activity:.4f}",
                    f"{r.activity_std:.4f}",
                    f"{r.final_avg_weight:.4f}",
                ]
            )

    print(f"\nResults saved to: {output_path}")

    # Save all avalanche data to single CSV
    avalanche_path = Path("output/soc_sweep_avalanches.csv")
    with open(avalanche_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "leak_rate",
                "reset_potential",
                "decay_alpha",
                "oja_alpha",
                "threshold",
                "size",
                "duration",
            ]
        )
        for row in all_avalanches:
            writer.writerow(row)

    print(f"Avalanche data saved to: {avalanche_path}")
    print(f"Total avalanches collected: {len(all_avalanches)}")

    # Find best candidates (closest to target metrics)
    print("\n=== TOP CANDIDATES (closest to BR=1.0 and slope=-1.5) ===")

    # Score: minimize distance from targets
    def score(r: SweepResult) -> float:
        br_dist = abs(r.branching_ratio - 1.0)
        slope_dist = abs(r.power_law_slope - (-1.5))
        # Penalize if no avalanches or dead activity
        if r.n_avalanches < 5 or r.avg_activity < 0.01:
            return float("inf")
        return br_dist + slope_dist

    sorted_results = sorted(results, key=score)

    for i, r in enumerate(sorted_results[:10]):
        print(
            f"{i + 1}. leak={r.leak_rate:.2f}, reset={r.reset_potential:.1f}, "
            f"decay={r.decay_alpha:.4f}, oja={r.oja_alpha:.3f}, thresh={r.threshold:.2f}"
        )
        print(
            f"   BR={r.branching_ratio:.3f}, slope={r.power_law_slope:.2f}, "
            f"avalanches={r.n_avalanches}, activity={r.avg_activity:.3f}"
        )


if __name__ == "__main__":
    main()
