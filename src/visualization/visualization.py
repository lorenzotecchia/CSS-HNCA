"""Pygame visualization for neural cellular automata.

Renders a projected 3D network with directed edges weighted by opacity.
Controls:
- Space: single step
- Enter: run/stop
- R: reset with new reproducible seed
- Esc: quit
- Left mouse drag: orbit camera
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from src.config.loader import load_config
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner

DEFAULT_CONFIG_PATH = Path("config/default.toml")
WINDOW_SIZE = (1500, 800)

PANEL_BG = (235, 235, 235)
BG = (245, 245, 245)
TEXT = (20, 20, 20)

NODE_FIRE = (220, 40, 40)
NODE_OFF = (0, 0, 0)
NODE_OUTLINE = (60, 60, 60)

EDGE_MIN_ALPHA = 255
EDGE_MAX_ALPHA = 255
EDGE_THICKNESS = 2
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
        # Draw label
        label_surf = font.render(self.label, True, TEXT)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))

        # Draw bar
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)

        # Draw fill
        if self.max_val > self.min_val:
            fill_w = ((self.val - self.min_val) / (self.max_val - self.min_val)) * self.rect.w
        else:
            fill_w = self.rect.w
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pygame.draw.rect(screen, (0, 150, 0), fill_rect)

        # Draw knob
        knob_x = self.rect.x + fill_w
        pygame.draw.circle(screen, (255, 0, 0), (int(knob_x), self.rect.centery), 5)

        # Draw value
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
        # Draw box
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        if self.val:
            pygame.draw.rect(screen, (0, 150, 0), self.rect.inflate(-4, -4))

        # Draw label
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
        # Draw label
        label_surf = font.render(self.label, True, TEXT)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))

        # Draw box
        color = (0, 150, 0) if self.active else (100, 100, 100)
        pygame.draw.rect(screen, color, self.rect, 2)

        # Draw text
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
    base = 200
    shade = int(base * (1.0 - u))
    r = shade
    g = shade
    b = shade
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


def create_simulation(config_path: Path, seed: int | None, n_neurons: int, k_prop: float, a: float, b: float) -> Simulation:
    config = load_config(config_path)

    network = Network.create_beta_weighted_directed(
        n_neurons=n_neurons,
        k_prop=k_prop,
        a=a,
        b=b,
        seed=seed,
    )

    state = NeuronState.create(
        n_neurons=n_neurons,
        threshold=config.learning.threshold,
        initial_firing_fraction=config.network.initial_firing_fraction,
        seed=seed,
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


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    base_seed = config.seed if config.seed is not None else int(time.time())
    reset_index = 0
    current_seed = base_seed

    # Initial network parameters
    n_neurons = config.network.n_neurons
    k_prop = 0.05  # Default
    beta_a = 2.0
    beta_b = 6.0

    simulation = create_simulation(DEFAULT_CONFIG_PATH, current_seed, n_neurons, k_prop, beta_a, beta_b)

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

    # UI Elements
    ui_elements = []
    y_pos = 100
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.1, simulation.learner.learning_rate, "L Rate"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.1, simulation.learner.forgetting_rate, "F Rate"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 1.0, simulation.learner.weight_min, "W Min"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 1.0, simulation.learner.weight_max, "W Max"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.01, simulation.learner.decay_alpha, "Decay"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.0, 0.01, simulation.learner.oja_alpha, "Oja"))
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
    y_pos += 50  # Space for network params
    ui_elements.append(TextInput(pygame.Rect(20, y_pos, 100, 25), str(n_neurons), "N Neurons"))
    y_pos += 35
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.001, 0.5, k_prop, "K Prop"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.1, 10.0, beta_a, "Beta A"))
    y_pos += 30
    ui_elements.append(Slider(pygame.Rect(20, y_pos, 200, 20), 0.1, 10.0, beta_b, "Beta B"))

    # Set initial k_prop bounds
    ui_elements[12].min_val = 2 / n_neurons
    ui_elements[12].max_val = 1 - 1 / n_neurons
    ui_elements[12].val = max(ui_elements[12].min_val, min(ui_elements[12].max_val, k_prop))

    def do_step() -> None:
        simulation.step()

    def do_reset() -> None:
        nonlocal reset_index, current_seed, projected_nodes, edge_indices, running_sim, accumulator, simulation, n_neurons, k_prop, beta_a, beta_b
        reset_index += 1
        current_seed = base_seed + reset_index
        # Update params from sliders
        try:
            n_neurons = int(ui_elements[11].text)
        except ValueError:
            pass  # keep old
        k_prop = ui_elements[12].val
        beta_a = ui_elements[13].val
        beta_b = ui_elements[14].val
        simulation = create_simulation(DEFAULT_CONFIG_PATH, current_seed, n_neurons, k_prop, beta_a, beta_b)
        projected_nodes = project_positions(
            simulation.network.positions, simulation.network.box_size, viewport, yaw, pitch, zoom
        )
        edge_indices = [tuple(idx) for idx in np.argwhere(simulation.network.link_matrix)]
        running_sim = False
        accumulator = 0.0

    def toggle_run() -> None:
        nonlocal running_sim
        running_sim = not running_sim

    alive = True
    while alive:
        dt = clock.tick(config.visualization.fps) / 1000.0

        for event in pygame.event.get():
            # Handle UI events first
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
                if event.key == pygame.K_ESCAPE:
                    alive = False
                elif event.key == pygame.K_SPACE:
                    do_step()
                elif event.key == pygame.K_RETURN:
                    toggle_run()
                elif event.key == pygame.K_r:
                    do_reset()

        # Update learner parameters from UI
        simulation.learner.learning_rate = ui_elements[0].val
        simulation.learner.forgetting_rate = ui_elements[1].val
        simulation.learner.weight_min = ui_elements[2].val
        simulation.learner.weight_max = ui_elements[3].val
        simulation.learner.decay_alpha = ui_elements[4].val
        simulation.learner.oja_alpha = ui_elements[5].val
        simulation.learner.enable_homeostatic = ui_elements[6].val
        old_timespan = simulation.learner.spike_timespan
        try:
            simulation.learner.spike_timespan = int(ui_elements[7].text)
        except ValueError:
            simulation.learner.spike_timespan = old_timespan
        if simulation.learner.spike_timespan != old_timespan:
            simulation.learner.spike_history = deque(simulation.learner.spike_history, maxlen=simulation.learner.spike_timespan)
        try:
            simulation.learner.min_spike_amount = int(ui_elements[8].text)
        except ValueError:
            pass
        try:
            simulation.learner.max_spike_amount = int(ui_elements[9].text)
        except ValueError:
            pass
        simulation.learner.weight_change_constant = ui_elements[10].val

        # Update k_prop slider bounds based on current n_neurons text
        try:
            current_n = int(ui_elements[11].text)
            if current_n >= 3:
                ui_elements[12].min_val = 2 / current_n
                ui_elements[12].max_val = 1 - 1 / current_n
                ui_elements[12].val = max(ui_elements[12].min_val, min(ui_elements[12].max_val, ui_elements[12].val))
        except ValueError:
            pass

        if running_sim:
            accumulator += dt
            step_dt = 1.0 / max(0.1, steps_per_second)
            while accumulator >= step_dt:
                simulation.step()
                accumulator -= step_dt

        screen.fill(BG)

        pygame.draw.rect(screen, PANEL_BG, pygame.Rect(0, 0, panel_w, screen.get_height()))
        pygame.draw.line(
            screen, (180, 180, 180), (panel_w, 0), (panel_w, screen.get_height()), 2
        )

        # Draw UI elements
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

        line_y = 56
        for line in [
            f"t = {simulation.time_step}",
            f"firing = {simulation.firing_count}",
            f"avg_w = {simulation.average_weight:.4f}",
            f"seed = {current_seed}",
            f"mode = {'run' if running_sim else 'paused'}",
        ]:
            txt = font.render(line, True, TEXT)
            screen.blit(txt, (20, line_y))
            line_y += 24

        hint_lines = [
            "Space = Step",
            "Enter = Run/Stop",
            "R = Reset",
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
    main()
