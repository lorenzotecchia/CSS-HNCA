#!/usr/bin/env python3
"""Render docs/assets/simulation.gif for the README: rotating 3D network + firing trace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner

# ponytail: README defaults hardcoded; switch to load_config once config/default.toml parses again
N, SEED, STEPS = 300, 42, 200
OUT = Path("docs/assets/simulation.gif")

network = Network.create_random(n_neurons=N, box_size=(10.0, 10.0, 10.0), radius=2.0,
                                initial_weight=0.06, seed=SEED, excitatory_fraction=1.0)
state = NeuronState.create(n_neurons=N, threshold=0.4, initial_firing_fraction=0.15,
                           seed=SEED, leak_rate=0.08, reset_potential=0.4)
learner = HebbianLearner(learning_rate=0.01, forgetting_rate=0.01, weight_min=0.0,
                         weight_max=0.3, decay_alpha=0.0005, oja_alpha=0.002)
sim = Simulation(network=network, state=state, learning_rate=0.01,
                 forgetting_rate=0.01, learner=learner)
sim.start()

fig = plt.figure(figsize=(10, 4.5), dpi=80)
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_ts = fig.add_subplot(1, 2, 2)
pos = network.positions
scatter = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=12, depthshade=False)
ax3d.set_axis_off()
(line,) = ax_ts.plot([], [], color="crimson", lw=1.5)
ax_ts.set(xlim=(0, STEPS), ylim=(0, N), xlabel="Time step", ylabel="Firing neurons",
          title="Network activity")
ax_ts.grid(alpha=0.3)
fig.suptitle("CSS-HNCA: LIF neurons with STDP")
fig.tight_layout()
counts = []


def frame(t):
    sim.step()
    counts.append(sim.firing_count)
    scatter.set_color(["crimson" if f else "0.75" for f in sim.state.firing])
    ax3d.view_init(elev=20, azim=t * 1.2)
    ax3d.set_title(f"t = {sim.time_step}   firing = {sim.firing_count}")
    line.set_data(range(len(counts)), counts)
    return scatter, line


OUT.parent.mkdir(parents=True, exist_ok=True)
FuncAnimation(fig, frame, frames=STEPS).save(OUT, writer=PillowWriter(fps=15))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
