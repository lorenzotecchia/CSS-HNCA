"""Pygame visualization for neural cellular automata.

Renders a projected 3D network with directed edges weighted by opacity.
Controls:
- Space: single step
- Enter: run/stop (apply input when editing)
- R: reset with new seed
- S: edit seed
- N: edit node count
- A: edit avalanche target
- Esc: quit (cancel input when editing)
- Left mouse drag: orbit camera
"""

from __future__ import annotations

import math
import time
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.weight_update import WeightUpdater
from src.visualization.avalanche_controller import AvalancheController

DEFAULT_CONFIG_PATH = Path("config/default.toml")
WINDOW_SIZE = (1500, 800)

PANEL_BG = (235, 235, 235)
BG = (245, 245, 245)
TEXT = (20, 20, 20)

NODE_FIRE = (220, 40, 40)
NODE_OFF = (0, 0, 0)
NODE_OUTLINE = (60, 60, 60)

EDGE_MIN_ALPHA = 40
EDGE_MAX_ALPHA = 160
EDGE_THICKNESS = 1
NODE_RADIUS = 8
ROTATE_SENSITIVITY = 0.006
ZOOM_SENSITIVITY = 0.1
ZOOM_MIN = 0.4
ZOOM_MAX = 2.5


@dataclass
class Slider:
    rect: pygame.Rect
    min_val: float
    max_val: float
    val: float
    label: str
    dragging: bool = False

    def draw(self, screen, font):
        label_surf = font.render(self.label, True, TEXT)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))

        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)

        if self.max_val > self.min_val:
            fill_w = ((self.val - self.min_val) / (self.max_val - self.min_val)) * self.rect.w
        else:
            fill_w = self.rect.w
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pygame.draw.rect(screen, (0, 150, 0), fill_rect)

        knob_x = self.rect.x + fill_w
        pygame.draw.circle(screen, (255, 0, 0), (int(knob_x), self.rect.centery), 5)

        val_surf = font.render(f"{self.val:.3f}", True, TEXT)
        screen.blit(val_surf, (self.rect.right + 10, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            frac = max(0.0, min(1.0, rel_x / self.rect.w))
            self.val = self.min_val + frac * (self.max_val - self.min_val)


@dataclass
class Checkbox:
    rect: pygame.Rect
    val: bool
    label: str

    def draw(self, screen, font):
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        if self.val:
            pygame.draw.rect(screen, (0, 150, 0), self.rect.inflate(-4, -4))

        label_surf = font.render(self.label, True, TEXT)
        screen.blit(label_surf, (self.rect.right + 10, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.val = not self.val


@dataclass
class TextInput:
    rect: pygame.Rect
    text: str
    label: str
    active: bool = False

    def draw(self, screen, font):
        label_surf = font.render(self.label, True, TEXT)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))

        color = (0, 150, 0) if self.active else (100, 100, 100)
        pygame.draw.rect(screen, color, self.rect, 2)

        text_surf = font.render(self.text, True, TEXT)
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode


@dataclass
class ProjectedNode:
    pos: pygame.Vector2
    depth: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def weight_to_color(weight: float, weight_min: float, weight_max: float) -> tuple[int, int, int, int]:
    if weight_max <= weight_min:
        u = 0.5
    else:
        u = (weight - weight_min) / (weight_max - weight_min)
    u = clamp01(u)

    alpha = int(EDGE_MIN_ALPHA + u * (EDGE_MAX_ALPHA - EDGE_MIN_ALPHA))
    low = (40, 80, 200)
    mid = (40, 180, 160)
    high = (240, 220, 80)
    if u < 0.5:
        t = u / 0.5
        r = int(low[0] + (mid[0] - low[0]) * t)
        g = int(low[1] + (mid[1] - low[1]) * t)
        b = int(low[2] + (mid[2] - low[2]) * t)
    else:
        t = (u - 0.5) / 0.5
        r = int(mid[0] + (high[0] - mid[0]) * t)
        g = int(mid[1] + (high[1] - mid[1]) * t)
        b = int(mid[2] + (high[2] - mid[2]) * t)
    return (r, g, b, alpha)


def project_positions(
    positions: np.ndarray,
    box_size: tuple[float, float, float],
    viewport: pygame.Rect,
    yaw: float,
    pitch: float,
    zoom: float,
) -> list[ProjectedNode]:
    center_box = np.array(box_size, dtype=float) / 2.0
    center_screen = pygame.Vector2(viewport.centerx, viewport.centery)

    scale = min(viewport.width, viewport.height) * 0.45 / max(box_size)
    scale *= zoom
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    projected: list[ProjectedNode] = []
    for pos in positions:
        x, y, z = pos - center_box
        xz = x * cos_yaw + z * sin_yaw
        zz = -x * sin_yaw + z * cos_yaw
        yz = y * cos_pitch - zz * sin_pitch
        zz = y * sin_pitch + zz * cos_pitch

        screen_x = center_screen.x + xz * scale
        screen_y = center_screen.y + yz * scale

        projected.append(ProjectedNode(pygame.Vector2(screen_x, screen_y), float(zz)))

    return projected


def draw_arrowhead(
    surface: pygame.Surface,
    start: pygame.Vector2,
    end: pygame.Vector2,
    color: tuple[int, int, int, int],
    thickness: int,
) -> None:
    direction = end - start
    length = direction.length()
    if length <= 1.0:
        return

    direction.scale_to_length(1.0)
    perp = pygame.Vector2(-direction.y, direction.x)
    arrow_len = 6 + thickness
    arrow_width = 4 + thickness

    tip = end
    base = end - direction * arrow_len
    left = base + perp * arrow_width
    right = base - perp * arrow_width

    pygame.draw.polygon(surface, color, [tip, left, right])


def create_simulation(
    config: "SimulationConfig",
    seed: int | None,
    n_neurons: int,
    k_prop: float,
    a: float,
    b: float,
    inhibitory_proportion: float,
    firing_count: int,
) -> Simulation:
    network = Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=k_prop,
        a=a,
        b=b,
        inhibitory_proportion=inhibitory_proportion,
        seed=seed,
    )

    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=config.learning.threshold,
        firing_count=firing_count,
        seed=seed,
        leak_rate=config.network.leak_rate,
        reset_potential=config.network.reset_potential,
    )

    learner = WeightUpdater(
        enable_stdp=True,
        enable_oja=False,
        enable_homeostatic=False,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        oja_alpha=config.learning.oja_alpha,
        spike_timespan=100,
        min_spike_amount=5,
        max_spike_amount=15,
        weight_change_constant=0.01,
    )

    return Simulation(
        network=network,
        state=state,
        learning_rate=config.learning.learning_rate,
        forgetting_rate=config.learning.forgetting_rate,
        learner=learner,
    )


def main(firing_count: int | None = None) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    base_seed = config.seed if config.seed is not None else int(time.time())
    reset_index = 0
    current_seed = base_seed

    n_neurons = config.network.n_neurons
    k_prop = 0.05
    beta_a = 2.0
    beta_b = 6.0
    inhibitory_proportion = 0.0
    stimulus_count = 1

    firing_count = firing_count if firing_count is not None else config.network.firing_count

    simulation = create_simulation(config, current_seed, n_neurons, k_prop, beta_a, beta_b, inhibitory_proportion, firing_count)
    avalanche_controller = AvalancheController(simulation, n_neurons=n_neurons, stimulus_count=stimulus_count)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Neural Network Evolution (Pygame)")
    clock = pygame.time.Clock()

    panel_w = 260
    viewport = pygame.Rect(panel_w, 0, WINDOW_SIZE[0] - panel_w, WINDOW_SIZE[1])

    yaw = 0.0
    pitch = 0.0
    zoom = 1.0
    projected_nodes = project_positions(
        simulation.network.positions, simulation.network.box_size, viewport, yaw, pitch, zoom
    )
    edge_indices = [tuple(idx) for idx in np.argwhere(simulation.network.link_matrix)]

    running_sim = False
    steps_per_second = 8.0
    accumulator = 0.0
    dragging = False
    last_mouse = pygame.Vector2(0, 0)

    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 28)

    edge_layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    ui_elements = []
    y_pos = 100
    ui_elements.append(Checkbox(pygame.Rect(20, y_pos, 20, 20), simulation.learner.enable_stdp, "STDP"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.1, simulation.learner.learning_rate, "L Rate"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.1, simulation.learner.forgetting_rate, "F Rate"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.01, simulation.learner.decay_alpha, "Decay"))
    y_pos += 30
    ui_elements.append(Checkbox(pygame.Rect(20, y_pos, 20, 20), simulation.learner.enable_oja, "Oja"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.01, simulation.learner.oja_alpha, "Oja Alpha"))
    y_pos += 30
    ui_elements.append(Checkbox(pygame.Rect(20, y_pos, 20, 20), simulation.learner.enable_homeostatic, "Homeo"))
    y_pos += 30
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(int(simulation.learner.spike_timespan)), "Timespan"))
    y_pos += 35
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(int(simulation.learner.min_spike_amount)), "Min Spikes"))
    y_pos += 35
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(int(simulation.learner.max_spike_amount)), "Max Spikes"))
    y_pos += 35
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.1, simulation.learner.weight_change_constant, "W Change"))
    y_pos += 50
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(n_neurons), "N Neurons"))
    y_pos += 35
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.001, 0.5, k_prop, "K Prop"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.1, 10.0, beta_a, "Beta A"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.1, 10.0, beta_b, "Beta B"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 1.0, inhibitory_proportion, "Inhib Prop"))
    y_pos += 30
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(stimulus_count), "Stimulus Count"))

    ui_elements[12].min_val = 2 / n_neurons
    ui_elements[12].max_val = 1 - 1 / n_neurons
    ui_elements[12].val = max(ui_elements[12].min_val, min(ui_elements[12].max_val, k_prop))

    input_mode: str | None = None
    input_buffer = ""
    input_error: str | None = None

    def read_network_params_from_ui() -> None:
        nonlocal n_neurons, k_prop, beta_a, beta_b, inhibitory_proportion, stimulus_count
        try:
            n_neurons = int(ui_elements[11].text)
        except ValueError:
            pass
        k_prop = ui_elements[12].val
        beta_a = ui_elements[13].val
        beta_b = ui_elements[14].val
        inhibitory_proportion = round(ui_elements[15].val, 2)  # Round to avoid rounding errors
        ui_elements[15].val = inhibitory_proportion  # Sync back to slider
        try:
            stimulus_count = int(ui_elements[16].text)
        except ValueError:
            pass

    def rebuild_simulation(new_seed: int | None = None) -> None:
        nonlocal simulation, projected_nodes, edge_indices, running_sim, accumulator, current_seed
        if new_seed is not None:
            current_seed = new_seed
        simulation = create_simulation(config, current_seed, n_neurons, k_prop, beta_a, beta_b, inhibitory_proportion, firing_count)
        projected_nodes = project_positions(
            simulation.network.positions, simulation.network.box_size, viewport, yaw, pitch, zoom
        )
        edge_indices = [tuple(idx) for idx in np.argwhere(simulation.network.link_matrix)]
        running_sim = False
        accumulator = 0.0
        avalanche_controller.rebind(simulation, n_neurons=n_neurons, stimulus_count=stimulus_count, reset_seen=True)

    def do_step() -> None:
        nonlocal running_sim
        simulation.step()
        if not avalanche_controller.record_step(simulation.time_step, simulation.firing_count):
            running_sim = False

    def do_reset() -> None:
        nonlocal reset_index, current_seed
        reset_index += 1
        current_seed = base_seed + reset_index
        read_network_params_from_ui()
        rebuild_simulation(new_seed=current_seed)

    def start_input(mode: str) -> None:
        nonlocal input_mode, input_buffer, input_error
        input_mode = mode
        input_error = None
        if mode == "seed":
            input_buffer = str(current_seed)
        elif mode == "nodes":
            input_buffer = str(n_neurons)
        else:
            input_buffer = str(avalanche_controller.target or "")

    def commit_input() -> None:
        nonlocal base_seed, reset_index, current_seed, n_neurons, input_mode, input_error
        nonlocal running_sim
        if input_mode is None:
            return
        try:
            value = int(input_buffer)
        except ValueError:
            input_error = "Enter a whole number."
            return
        if input_mode == "seed":
            base_seed = value
            reset_index = 0
            current_seed = value
            rebuild_simulation(new_seed=current_seed)
        elif input_mode == "nodes":
            if value <= 0:
                input_error = "Node count must be > 0."
                return
            n_neurons = value
            ui_elements[11].text = str(n_neurons)
            read_network_params_from_ui()
            rebuild_simulation(new_seed=current_seed)
        elif input_mode == "avalanches":
            try:
                avalanche_controller.set_target(value)
            except ValueError as exc:
                input_error = str(exc)
                return
            running_sim = True
        input_mode = None
        input_error = None

    def toggle_run() -> None:
        nonlocal running_sim
        running_sim = not running_sim

    alive = True
    while alive:
        dt = clock.tick(config.visualization.fps) / 1000.0

        for event in pygame.event.get():
            for elem in ui_elements:
                elem.handle_event(event)

            if event.type == pygame.QUIT:
                alive = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse = pygame.Vector2(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                zoom *= 1.0 + (event.y * ZOOM_SENSITIVITY)
                zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
                projected_nodes = project_positions(
                    simulation.network.positions,
                    simulation.network.box_size,
                    viewport,
                    yaw,
                    pitch,
                    zoom,
                )
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                pos = pygame.Vector2(event.pos)
                delta = pos - last_mouse
                last_mouse = pos
                yaw += delta.x * ROTATE_SENSITIVITY
                pitch += delta.y * ROTATE_SENSITIVITY
                pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch))
                projected_nodes = project_positions(
                    simulation.network.positions,
                    simulation.network.box_size,
                    viewport,
                    yaw,
                    pitch,
                    zoom,
                )
            elif event.type == pygame.KEYDOWN:
                if input_mode is not None:
                    if event.key == pygame.K_ESCAPE:
                        input_mode = None
                        input_error = None
                    elif event.key == pygame.K_RETURN:
                        commit_input()
                    elif event.key == pygame.K_BACKSPACE:
                        input_buffer = input_buffer[:-1]
                    else:
                        char = event.unicode
                        if char.isdigit() or (char == "-" and not input_buffer):
                            input_buffer += char
                else:
                    if event.key == pygame.K_ESCAPE:
                        alive = False
                    elif event.key == pygame.K_SPACE:
                        do_step()
                    elif event.key == pygame.K_RETURN:
                        toggle_run()
                    elif event.key == pygame.K_r:
                        do_reset()
                    elif event.key == pygame.K_s:
                        start_input("seed")
                    elif event.key == pygame.K_n:
                        start_input("nodes")
                    elif event.key == pygame.K_a:
                        start_input("avalanches")

        simulation.learner.enable_stdp = ui_elements[0].val
        simulation.learner.learning_rate = ui_elements[1].val
        simulation.learner.forgetting_rate = ui_elements[2].val
        simulation.learner.decay_alpha = ui_elements[3].val
        simulation.learner.enable_oja = ui_elements[4].val
        simulation.learner.oja_alpha = ui_elements[5].val
        simulation.learner.enable_homeostatic = ui_elements[6].val
        old_timespan = simulation.learner.spike_timespan
        try:
            simulation.learner.spike_timespan = int(ui_elements[7].text)
        except ValueError:
            simulation.learner.spike_timespan = old_timespan
        if simulation.learner.spike_timespan != old_timespan:
            simulation.learner.spike_history = deque(
                simulation.learner.spike_history, maxlen=simulation.learner.spike_timespan
            )
        try:
            simulation.learner.min_spike_amount = int(ui_elements[8].text)
        except ValueError:
            pass
        try:
            simulation.learner.max_spike_amount = int(ui_elements[9].text)
        except ValueError:
            pass
        simulation.learner.weight_change_constant = ui_elements[10].val

        try:
            current_n = int(ui_elements[11].text)
            if current_n >= 3:
                ui_elements[12].min_val = 2 / current_n
                ui_elements[12].max_val = 1 - 1 / current_n
                ui_elements[12].val = max(
                    ui_elements[12].min_val, min(ui_elements[12].max_val, ui_elements[12].val)
                )
        except ValueError:
            pass

        # Update avalanche stimulus count
        try:
            avalanche_controller.stimulus_count = int(ui_elements[16].text)
            avalanche_controller.stimulus_count = max(1, min(avalanche_controller.stimulus_count, n_neurons))
        except ValueError:
            pass

        if running_sim:
            accumulator += dt
            step_dt = 1.0 / max(0.1, steps_per_second)
            while accumulator >= step_dt:
                do_step()
                accumulator -= step_dt
            avalanche_controller.update(dt)

        screen.fill(BG)

        pygame.draw.rect(screen, PANEL_BG, pygame.Rect(0, 0, panel_w, screen.get_height()))
        pygame.draw.line(
            screen, (180, 180, 180), (panel_w, 0), (panel_w, screen.get_height()), 2
        )

        for elem in ui_elements:
            elem.draw(screen, font)

        edge_layer.fill((0, 0, 0, 0))
        linked_weights = simulation.network.weight_matrix[simulation.network.link_matrix]
        if linked_weights.size > 0:
            w_min = float(np.min(linked_weights))
            w_max = float(np.max(linked_weights))
        else:
            w_min = config.network.weight_min
            w_max = config.network.weight_max
        if w_max <= w_min:
            w_min -= 1e-6
            w_max += 1e-6
        for a, b in edge_indices:
            weight = simulation.network.weight_matrix[a, b]
            color = weight_to_color(weight, w_min, w_max)
            start = projected_nodes[a].pos
            end = projected_nodes[b].pos
            pygame.draw.line(edge_layer, color, start, end, EDGE_THICKNESS)
            draw_arrowhead(edge_layer, start, end, color, EDGE_THICKNESS)
        screen.blit(edge_layer, (0, 0))

        draw_order = sorted(range(simulation.network.n_neurons), key=lambda i: projected_nodes[i].depth)
        for idx in draw_order:
            node = simulation.state.firing[idx]
            proj = projected_nodes[idx]
            radius = NODE_RADIUS
            fill = NODE_FIRE if node else NODE_OFF
            pygame.draw.circle(screen, fill, (int(proj.pos.x), int(proj.pos.y)), radius)
            pygame.draw.circle(
                screen, NODE_OUTLINE, (int(proj.pos.x), int(proj.pos.y)), radius, 2
            )

        title = font_big.render("Controls", True, TEXT)
        screen.blit(title, (20, 20))

        stats_lines = [
            f"t = {simulation.time_step}",
            f"firing = {simulation.firing_count}",
            f"avg_w = {simulation.average_weight:.4f}",
            f"seed = {current_seed}",
            f"nodes = {n_neurons}",
            avalanche_controller.status_line(),
            f"mode = {'run' if running_sim else 'paused'}",
        ]
        if input_mode is not None:
            label = {"seed": "seed", "nodes": "nodes", "avalanches": "avalanches"}[input_mode]
            input_line = f"{label} = {input_buffer or ''}_"
            target = 3 if input_mode == "seed" else 4 if input_mode == "nodes" else 5
            stats_lines[target] = input_line

        line_y = 56
        for line in stats_lines:
            txt = font.render(line, True, TEXT)
            screen.blit(txt, (20, line_y))
            line_y += 24
        if input_mode is not None and input_error:
            err = font.render(input_error, True, (180, 40, 40))
            screen.blit(err, (20, line_y + 8))

        hint_lines = [
            "Space = Step",
            "Enter = Run/Stop",
            "R = Reset",
            "S = Edit Seed",
            "N = Edit Nodes",
            "A = Edit Avalanches",
            "Wheel = Zoom",
            "Esc = Quit",
        ]
        hint_y = screen.get_height() - (len(hint_lines) * 20) - 20
        for line in hint_lines:
            hint = font.render(line, True, (60, 60, 60))
            screen.blit(hint, (20, hint_y))
            hint_y += 20

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pygame visualization for neural cellular automata")
    parser.add_argument("--firing-count", type=int, help="Number of neurons initially firing")
    args = parser.parse_args()
    main(firing_count=args.firing_count)
