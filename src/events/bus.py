"""Event bus for decoupled communication.

Provides typed event subscription and emission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class StepEvent:
    """Event emitted after each simulation step."""

    time_step: int
    firing_count: int
    avg_weight: float


@dataclass
class ResetEvent:
    """Event emitted when simulation is reset."""

    seed: int | None


class EventBus:
    """Typed event bus for publish/subscribe pattern.

    Supports multiple handlers per event type and type-safe filtering.
    """

    def __init__(self) -> None:
        """Initialize empty event bus."""
        self._handlers: dict[type, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe handler to event type.

        Args:
            event_type: The type of event to subscribe to
            handler: Callable that receives events of the given type
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Unsubscribe handler from event type.

        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def emit(self, event: T) -> None:
        """Emit event to all subscribed handlers.

        Args:
            event: The event to emit
        """
        event_type = type(event)
        for handler in self._handlers[event_type]:
            handler(event)
