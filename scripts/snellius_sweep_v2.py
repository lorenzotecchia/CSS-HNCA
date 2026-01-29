#!/usr/bin/env python3
"""Snellius Supercomputer Parameter Sweep v2.

Zoomed-in sweep around avalanche-producing regimes with excitatory_fraction
as a new dimension and a finer grid over Oja/decay regularisation parameters.

Runs a single grid configuration (identified by --config-index) with 500
2D Latin Hypercube samples over independent (learning_rate, forgetting_rate).

Designed for SLURM array jobs: each array task processes one of 1080 configs.

Features:
    - 2D LHS sampling: independent learning_rate and forgetting_rate (not diagonal)
    - Incremental checkpointing: saves partial results every CHECKPOINT_INTERVAL
    - Resume capability: loads existing checkpoint and skips completed samples
    - Graceful shutdown on SIGTERM/SIGINT: saves partial work before exit

Usage:
    python scripts/snellius_sweep_v2.py --config-index 0
    # or via SLURM:
    #SBATCH --array=0-1079
    python scripts/snellius_sweep_v2.py --config-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import atexit
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass, fields
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

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
# Parameter grid: 5 x 2 x 3 x 4 x 3 x 3 = 1,080 configurations
# ---------------------------------------------------------------------------

EXCITATORY_FRACTIONS = [0.75, 0.80, 0.85, 0.90, 1.0]
FIRING_FRACTIONS = [0.05, 0.10, 0.15]
# Only avalanche-producing leak/reset pairs from sweep v1
LEAK_RESET_PAIRS = [(0.15, 0.60), (1.00, 0.00)]
K_PROPS = [5 / 500, 10 / 500, 15 / 500]  # 0.01, 0.02, 0.03
# Fine grid around onset of regularisation
DECAY_ALPHAS = [0.0001, 0.0003, 0.0005, 0.001]
OJA_ALPHAS = [0.001, 0.002, 0.004]

GRID = list(product(
    EXCITATORY_FRACTIONS,
    FIRING_FRACTIONS,
    LEAK_RESET_PAIRS,
    K_PROPS,
    DECAY_ALPHAS,
    OJA_ALPHAS,
))
N_CONFIGS = len(GRID)  # 1080

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
N_NEURONS = 500
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
N_SAMPLES = 500
LR_BOUNDS = (0.00001, 0.1)

# Checkpoint settings
CHECKPOINT_INTERVAL = 50  # Save every N completed samples


# ---------------------------------------------------------------------------
# Checkpointing utilities
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT by setting shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nReceived signal {signum}, will save checkpoint and exit...", flush=True)


def save_checkpoint(
    results: list["SweepResult"],
    checkpoint_path: Path,
) -> None:
    """Save current results to checkpoint file."""
    if not results:
        return
    header = [f.name for f in fields(SweepResult)]
    with open(checkpoint_path, "w", newline="") as fh:
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
    print(f"  Checkpoint saved: {len(results)} results to {checkpoint_path}", flush=True)


def load_checkpoint(checkpoint_path: Path) -> tuple[list["SweepResult"], set[tuple[float, float]]]:
    """Load existing checkpoint if present. Returns (results, completed_rate_pairs)."""
    if not checkpoint_path.exists():
        return [], set()
    
    results: list[SweepResult] = []
    completed_rates: set[tuple[float, float]] = set()
    
    try:
        with open(checkpoint_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lr = float(row["learning_rate"])
                fr = float(row["forgetting_rate"])
                completed_rates.add((lr, fr))
                
                results.append(SweepResult(
                    n_neurons=int(row["n_neurons"]),
                    excitatory_fraction=float(row["excitatory_fraction"]),
                    firing_count=int(row["firing_count"]),
                    leak_rate=float(row["leak_rate"]),
                    reset_potential=float(row["reset_potential"]),
                    k_prop=float(row["k_prop"]),
                    decay_alpha=float(row["decay_alpha"]),
                    oja_alpha=float(row["oja_alpha"]),
                    learning_rate=float(row["learning_rate"]),
                    forgetting_rate=float(row["forgetting_rate"]),
                    avg_weight=float(row["avg_weight"]),
                    std_weight=float(row["std_weight"]),
                    avg_firing_count=float(row["avg_firing_count"]),
                    std_firing_count=float(row["std_firing_count"]),
                    avg_avalanche_size=float(row["avg_avalanche_size"]),
                    std_avalanche_size=float(row["std_avalanche_size"]),
                    avg_avalanche_duration=float(row["avg_avalanche_duration"]),
                    std_avalanche_duration=float(row["std_avalanche_duration"]),
                    n_avalanches=int(row["n_avalanches"]),
                ))
        print(f"  Loaded checkpoint: {len(results)} completed results", flush=True)
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}", flush=True)
        return [], set()
    
    return results, completed_rates


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    """Result of a single simulation run."""

    # Grid parameters
    n_neurons: int
    excitatory_fraction: float
    firing_count: int
    leak_rate: float
    reset_potential: float
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
    excitatory_fraction: float,
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
        excitatory_fraction=excitatory_fraction,
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
        excitatory_fraction=excitatory_fraction,
        firing_count=firing_count,
        leak_rate=leak_rate,
        reset_potential=reset_potential,
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
    (lr, fr, idx, n_total, excitatory_fraction, firing_count,
     leak_rate, reset_potential, k_prop, decay_alpha, oja_alpha) = args
    print(
        f"  [{idx}/{n_total}] lr={lr:.6f}, fr={fr:.6f}",
        flush=True,
    )
    try:
        return run_single(
            learning_rate=lr,
            forgetting_rate=fr,
            excitatory_fraction=excitatory_fraction,
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
    global _shutdown_requested
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    
    parser = argparse.ArgumentParser(description="Snellius parameter sweep v2")
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
    (excitatory_fraction, firing_frac, (leak_rate, reset_potential),
     k_prop, decay_alpha, oja_alpha) = GRID[idx]
    firing_count = int(firing_frac * N_NEURONS)

    print(f"Config {idx}/{N_CONFIGS - 1}:")
    print(f"  excitatory_fraction={excitatory_fraction}")
    print(f"  firing_count={firing_count} (frac={firing_frac})")
    print(f"  leak_rate={leak_rate}, reset_potential={reset_potential}")
    print(f"  k_prop={k_prop:.4f}")
    print(f"  decay_alpha={decay_alpha}, oja_alpha={oja_alpha}")
    print(f"  n_samples={N_SAMPLES}, steps={TOTAL_STEPS}, warmup={WARMUP_STEPS}")
    print()

    # Setup output paths
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_v2_config_{idx:04d}.csv"
    checkpoint_path = output_dir / f"sweep_v2_config_{idx:04d}.checkpoint.csv"
    
    # Load checkpoint if exists
    results, completed_rates = load_checkpoint(checkpoint_path)

    # 2D Latin Hypercube Sampling for independent (lr, fr)
    rng = np.random.default_rng(SEED)
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    samples = sampler.random(n=N_SAMPLES)
    learning_rates = LR_BOUNDS[0] + (LR_BOUNDS[1] - LR_BOUNDS[0]) * samples[:, 0]
    forgetting_rates = LR_BOUNDS[0] + (LR_BOUNDS[1] - LR_BOUNDS[0]) * samples[:, 1]

    # Filter out already-completed (lr, fr) pairs (using approximate matching)
    def pair_completed(lr: float, fr: float) -> bool:
        return any(abs(lr - clr) < 1e-10 and abs(fr - cfr) < 1e-10 
                   for clr, cfr in completed_rates)
    
    pending_tasks = [
        (lr, fr, i + 1, N_SAMPLES, excitatory_fraction, firing_count,
         leak_rate, reset_potential, k_prop, decay_alpha, oja_alpha)
        for i, (lr, fr) in enumerate(zip(learning_rates, forgetting_rates))
        if not pair_completed(lr, fr)
    ]
    
    if not pending_tasks:
        print(f"All {N_SAMPLES} samples already completed!")
        if checkpoint_path.exists():
            import shutil
            shutil.copy(checkpoint_path, output_path)
            checkpoint_path.unlink()
        print(f"Results saved to: {output_path} ({len(results)} rows)")
        return

    # Detect available CPUs — prefer SLURM allocation over cpu_count()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        n_workers = int(slurm_cpus)
        print(f"SLURM_CPUS_PER_TASK={slurm_cpus}")
    else:
        try:
            n_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            n_workers = cpu_count() or 1
    n_workers = max(1, n_workers)
    print(f"Running {len(pending_tasks)} pending samples with {n_workers} workers...")
    print(f"  (Checkpoint interval: every {CHECKPOINT_INTERVAL} samples)")
    t_start = time.monotonic()

    # Process in batches for checkpointing
    batch_size = min(CHECKPOINT_INTERVAL, len(pending_tasks))
    
    def save_and_exit() -> None:
        """Save checkpoint on exit."""
        save_checkpoint(results, checkpoint_path)
    
    atexit.register(save_and_exit)

    with Pool(processes=n_workers) as pool:
        for batch_start in range(0, len(pending_tasks), batch_size):
            if _shutdown_requested:
                print("Shutdown requested, saving checkpoint...", flush=True)
                break
            
            batch = pending_tasks[batch_start:batch_start + batch_size]
            batch_results = pool.map(_worker, batch)
            
            # Collect successful results
            for r in batch_results:
                if r is not None:
                    results.append(r)
            
            # Save checkpoint after each batch
            save_checkpoint(results, checkpoint_path)

    atexit.unregister(save_and_exit)
    elapsed = time.monotonic() - t_start
    print(f"\nCompleted {len(results)}/{N_SAMPLES} runs in {elapsed:.1f}s")

    # Write final CSV
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

    # Remove checkpoint on successful completion
    if checkpoint_path.exists() and len(results) == N_SAMPLES:
        checkpoint_path.unlink()
        
    print(f"Results saved to: {output_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
