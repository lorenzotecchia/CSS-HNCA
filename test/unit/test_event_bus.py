"""Unit tests for EventBus.

Tests subscribe, emit, multiple handlers, and type filtering.
"""

from dataclasses import dataclass


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


class TestEventBusSubscribeAndEmit:
    """Test basic subscribe and emit functionality."""

    def test_handler_receives_emitted_event(self) -> None:
        """Subscribed handler should receive emitted event."""
        from src.events.bus import EventBus

        bus = EventBus()
        received: list[StepEvent] = []

        def handler(event: StepEvent) -> None:
            received.append(event)

        bus.subscribe(StepEvent, handler)
        event = StepEvent(time_step=1, firing_count=10, avg_weight=0.5)
        bus.emit(event)

        assert len(received) == 1
        assert received[0] == event

    def test_unsubscribed_handler_does_not_receive_event(self) -> None:
        """Handler not subscribed should not receive events."""
        from src.events.bus import EventBus

        bus = EventBus()
        received: list[StepEvent] = []

        def handler(event: StepEvent) -> None:
            received.append(event)

        # Do NOT subscribe handler
        bus.emit(StepEvent(time_step=1, firing_count=10, avg_weight=0.5))

        assert len(received) == 0


class TestEventBusMultipleHandlers:
    """Test multiple handlers for same event type."""

    def test_multiple_handlers_all_receive_event(self) -> None:
        """All subscribed handlers should receive emitted event."""
        from src.events.bus import EventBus

        bus = EventBus()
        received_1: list[StepEvent] = []
        received_2: list[StepEvent] = []

        def handler_1(event: StepEvent) -> None:
            received_1.append(event)

        def handler_2(event: StepEvent) -> None:
            received_2.append(event)

        bus.subscribe(StepEvent, handler_1)
        bus.subscribe(StepEvent, handler_2)

        event = StepEvent(time_step=5, firing_count=20, avg_weight=0.3)
        bus.emit(event)

        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0] == event
        assert received_2[0] == event


class TestEventBusTypeFiltering:
    """Test that handlers only receive events of subscribed type."""

    def test_handler_only_receives_subscribed_event_type(self) -> None:
        """Handler subscribed to StepEvent should not receive ResetEvent."""
        from src.events.bus import EventBus

        bus = EventBus()
        step_events: list[StepEvent] = []
        reset_events: list[ResetEvent] = []

        def step_handler(event: StepEvent) -> None:
            step_events.append(event)

        def reset_handler(event: ResetEvent) -> None:
            reset_events.append(event)

        bus.subscribe(StepEvent, step_handler)
        bus.subscribe(ResetEvent, reset_handler)

        bus.emit(StepEvent(time_step=1, firing_count=10, avg_weight=0.5))
        bus.emit(ResetEvent(seed=42))

        assert len(step_events) == 1
        assert len(reset_events) == 1

    def test_emitting_unsubscribed_event_type_does_nothing(self) -> None:
        """Emitting event with no subscribers should not error."""
        from src.events.bus import EventBus

        bus = EventBus()
        # No subscribers for ResetEvent
        bus.emit(ResetEvent(seed=None))  # Should not raise


class TestEventBusUnsubscribe:
    """Test unsubscribe functionality."""

    def test_unsubscribed_handler_stops_receiving_events(self) -> None:
        """After unsubscribe, handler should not receive events."""
        from src.events.bus import EventBus

        bus = EventBus()
        received: list[StepEvent] = []

        def handler(event: StepEvent) -> None:
            received.append(event)

        bus.subscribe(StepEvent, handler)
        bus.emit(StepEvent(time_step=1, firing_count=10, avg_weight=0.5))
        assert len(received) == 1

        bus.unsubscribe(StepEvent, handler)
        bus.emit(StepEvent(time_step=2, firing_count=15, avg_weight=0.6))
        assert len(received) == 1  # Still 1, not 2


class TestEventBusMultipleEmits:
    """Test multiple emit calls."""

    def test_handler_receives_all_emitted_events(self) -> None:
        """Handler should receive all events emitted."""
        from src.events.bus import EventBus

        bus = EventBus()
        received: list[StepEvent] = []

        def handler(event: StepEvent) -> None:
            received.append(event)

        bus.subscribe(StepEvent, handler)

        for i in range(5):
            bus.emit(StepEvent(time_step=i, firing_count=i * 10, avg_weight=0.1 * i))

        assert len(received) == 5
        assert [e.time_step for e in received] == [0, 1, 2, 3, 4]
