#!/usr/bin/env python3
"""Snellius Supercomputer Parameter Sweep.

Runs a single grid configuration (identified by --config-index) with 3,000
Latin Hypercube samples along the diagonal learning_rate == forgetting_rate.

Designed for SLURM array jobs: each array task processes one of 54 configs.

Usage:
    python scripts/snellius_sweep.py --config-index 0
    # or via SLURM:
    #SBATCH --array=0-53
    python scripts/snellius_sweep.py --config-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector
from src.learning.weight_update import WeightUpdater

# ---------------------------------------------------------------------------
# Parameter grid: 3 x 3 x 3 x 2 = 54 configurations
# ---------------------------------------------------------------------------

FIRING_FRACTIONS = [0.05, 0.10, 0.15]  # -> firing_count = fraction * n_neurons
LEAK_RESET_PAIRS = [(0.0, 0.0), (1.0, 0.0), (0.15, 0.6)]
K_PROPS = [5 / 500, 10 / 500, 15 / 500]  # 0.01, 0.02, 0.03
DECAY_OJA_PAIRS = [(0.0, 0.0), (0.0005, 0.002)]

GRID = list(product(FIRING_FRACTIONS, LEAK_RESET_PAIRS, K_PROPS, DECAY_OJA_PAIRS))
N_CONFIGS = len(GRID)  # 54

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
N_NEURONS = 500
EXCITATORY_FRACTION = 1.0
THRESHOLD = 0.5
BETA_A = 2.0
BETA_B = 6.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
WEIGHT_MIN_INH = -1.0
WEIGHT_MAX_INH = 0.0
SEED = 42

TOTAL_STEPS = 8000
WARMUP_STEPS = 1000
N_SAMPLES = 3000
LR_BOUNDS = (0.00001, 0.1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    """Result of a single simulation run."""

    # Grid parameters
    n_neurons: int
    firing_count: int
    leak_rate: float
    reset_potential: float
    excitatory_fraction: float
    k_prop: float
    decay_alpha: float
    oja_alpha: float

    # Sampled parameters
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


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------
def run_single(
    learning_rate: float,
    forgetting_rate: float,
    firing_count: int,
    leak_rate: float,
    reset_potential: float,
    k_prop: float,
    decay_alpha: float,
    oja_alpha: float,
) -> SweepResult:
    """Run one simulation and return metrics."""
    network = Network.create_beta_weighted_directed(
        n_neurons=N_NEURONS,
        k_prop=k_prop,
        a=BETA_A,
        b=BETA_B,
        excitatory_fraction=EXCITATORY_FRACTION,
        weight_min=WEIGHT_MIN,
        weight_max=WEIGHT_MAX,
        weight_min_inh=WEIGHT_MIN_INH,
        weight_max_inh=WEIGHT_MAX_INH,
        seed=SEED,
    )

    state = NeuronState.create(
        n_neurons=N_NEURONS,
        threshold=THRESHOLD,
        firing_count=firing_count,
        seed=SEED,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
    )

    learner = WeightUpdater(
        enable_stdp=True,
        enable_oja=oja_alpha > 0,
        enable_homeostatic=False,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        weight_min=WEIGHT_MIN,
        weight_max=WEIGHT_MAX,
        decay_alpha=decay_alpha,
        oja_alpha=oja_alpha,
        weight_min_inh=WEIGHT_MIN_INH,
        weight_max_inh=WEIGHT_MAX_INH,
    )

    simulation = Simulation(
        network=network,
        state=state,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        learner=learner,
    )

    detector = AvalancheDetector(
        n_neurons=N_NEURONS,
        quiet_threshold=0.05,
    )

    weight_history: list[float] = []
    firing_history: list[int] = []

    simulation.start()
    restart_count = 0

    for step in range(TOTAL_STEPS):
        simulation.step()
        detector.record_step(simulation.time_step, simulation.firing_count)

        if step >= WARMUP_STEPS:
            weight_history.append(simulation.average_weight)
            firing_history.append(simulation.firing_count)

        if simulation.firing_count == 0:
            restart_count += 1
            simulation.state.reinitialize_firing(
                firing_fraction=firing_count / N_NEURONS,
                seed=SEED + restart_count,
            )

    detector.finalize()

    # Weight stats
    w = np.array(weight_history)
    avg_weight = float(np.mean(w)) if len(w) > 0 else 0.0
    std_weight = float(np.std(w)) if len(w) > 0 else 0.0

    # Firing stats
    f = np.array(firing_history)
    avg_firing = float(np.mean(f)) if len(f) > 0 else 0.0
    std_firing = float(np.std(f)) if len(f) > 0 else 0.0

    # Avalanche stats (only after warmup)
    post_warmup = [a for a in detector.avalanches if a.start_time >= WARMUP_STEPS]
    sizes = [a.size for a in post_warmup]
    durations = [a.duration for a in post_warmup]

    avg_aval_size = float(np.mean(sizes)) if sizes else 0.0
    std_aval_size = float(np.std(sizes)) if sizes else 0.0
    avg_aval_dur = float(np.mean(durations)) if durations else 0.0
    std_aval_dur = float(np.std(durations)) if durations else 0.0

    return SweepResult(
        n_neurons=N_NEURONS,
        firing_count=firing_count,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
        excitatory_fraction=EXCITATORY_FRACTION,
        k_prop=k_prop,
        decay_alpha=decay_alpha,
        oja_alpha=oja_alpha,
        learning_rate=learning_rate,
        forgetting_rate=forgetting_rate,
        avg_weight=avg_weight,
        std_weight=std_weight,
        avg_firing_count=avg_firing,
        std_firing_count=std_firing,
        avg_avalanche_size=avg_aval_size,
        std_avalanche_size=std_aval_size,
        avg_avalanche_duration=avg_aval_dur,
        std_avalanche_duration=std_aval_dur,
        n_avalanches=len(post_warmup),
    )


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> SweepResult | None:
    rate, idx, n_total, firing_count, leak_rate, reset_potential, k_prop, decay_alpha, oja_alpha = args
    print(
        f"  [{idx}/{n_total}] lr=fr={rate:.6f}",
        flush=True,
    )
    try:
        return run_single(
            learning_rate=rate,
            forgetting_rate=rate,
            firing_count=firing_count,
            leak_rate=leak_rate,
            reset_potential=reset_potential,
            k_prop=k_prop,
            decay_alpha=decay_alpha,
            oja_alpha=oja_alpha,
        )
    except Exception as e:
        print(f"  [{idx}/{n_total}] ERROR: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Snellius parameter sweep")
    parser.add_argument(
        "--config-index",
        type=int,
        required=True,
        help=f"Grid config index (0-{N_CONFIGS - 1})",
    )
    args = parser.parse_args()

    idx = args.config_index
    if idx < 0 or idx >= N_CONFIGS:
        print(f"Error: config-index must be 0-{N_CONFIGS - 1}, got {idx}")
        sys.exit(1)

    # Unpack grid parameters
    firing_frac, (leak_rate, reset_potential), k_prop, (decay_alpha, oja_alpha) = GRID[idx]
    firing_count = int(firing_frac * N_NEURONS)

    print(f"Config {idx}/{N_CONFIGS - 1}:")
    print(f"  firing_count={firing_count} (frac={firing_frac})")
    print(f"  leak_rate={leak_rate}, reset_potential={reset_potential}")
    print(f"  k_prop={k_prop:.4f}")
    print(f"  decay_alpha={decay_alpha}, oja_alpha={oja_alpha}")
    print(f"  n_samples={N_SAMPLES}, steps={TOTAL_STEPS}, warmup={WARMUP_STEPS}")
    print()

    # Latin Hypercube Sampling along diagonal (lr == fr)
    rng = np.random.default_rng(SEED)
    sampler = qmc.LatinHypercube(d=1, seed=rng)
    sample = sampler.random(n=N_SAMPLES).flatten()
    rates = LR_BOUNDS[0] + (LR_BOUNDS[1] - LR_BOUNDS[0]) * sample

    # Prepare worker arguments
    n_workers = max(1, cpu_count())
    print(f"Running {N_SAMPLES} samples with {n_workers} workers...")

    tasks = [
        (rate, i + 1, N_SAMPLES, firing_count, leak_rate, reset_potential, k_prop, decay_alpha, oja_alpha)
        for i, rate in enumerate(rates)
    ]

    with Pool(processes=n_workers) as pool:
        results_raw = pool.map(_worker, tasks)

    results: list[SweepResult] = [r for r in results_raw if r is not None]
    print(f"\nCompleted {len(results)}/{N_SAMPLES} runs")

    # Write CSV
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_config_{idx:03d}.csv"

    header = [f.name for f in fields(SweepResult)]
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in results:
            row = []
            for f in fields(SweepResult):
                val = getattr(r, f.name)
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(val)
            writer.writerow(row)

    print(f"Results saved to: {output_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
