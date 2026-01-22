"""Property tests for EventBus.

Tests that all subscribed handlers receive events.
"""

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st


@dataclass(frozen=True)
class SampleEvent:
    """Sample event for property testing."""

    value: int


class TestEventBusProperties:
    """Property-based tests for EventBus."""

    @given(st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_all_handlers_receive_all_events(self, event_values: list[int]) -> None:
        """All subscribed handlers receive all emitted events."""
        from src.events.bus import EventBus

        bus = EventBus()

        received_1: list[SampleEvent] = []
        received_2: list[SampleEvent] = []
        received_3: list[SampleEvent] = []

        bus.subscribe(SampleEvent, lambda e: received_1.append(e))
        bus.subscribe(SampleEvent, lambda e: received_2.append(e))
        bus.subscribe(SampleEvent, lambda e: received_3.append(e))

        for v in event_values:
            bus.emit(SampleEvent(value=v))

        # All handlers should have received all events
        assert len(received_1) == len(event_values)
        assert len(received_2) == len(event_values)
        assert len(received_3) == len(event_values)

        # Events should be in order
        assert [e.value for e in received_1] == event_values
        assert [e.value for e in received_2] == event_values
        assert [e.value for e in received_3] == event_values

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_handler_count_matches_subscription_count(self, n_handlers: int) -> None:
        """Number of handlers that receive event matches subscription count."""
        from src.events.bus import EventBus

        bus = EventBus()
        call_count = 0

        def make_handler() -> None:
            nonlocal call_count

            def handler(e: SampleEvent) -> None:
                nonlocal call_count
                call_count += 1

            return handler

        for _ in range(n_handlers):
            bus.subscribe(SampleEvent, make_handler())

        bus.emit(SampleEvent(value=42))

        assert call_count == n_handlers

    @given(
        st.lists(st.integers(), min_size=0, max_size=10),
        st.lists(st.integers(), min_size=0, max_size=10),
    )
    @settings(max_examples=30)
    def test_type_isolation(self, type_a_values: list[int], type_b_values: list[int]) -> None:
        """Events of different types are isolated to their handlers."""
        from src.events.bus import EventBus

        @dataclass(frozen=True)
        class TypeAEvent:
            value: int

        @dataclass(frozen=True)
        class TypeBEvent:
            value: int

        bus = EventBus()
        type_a_received: list[TypeAEvent] = []
        type_b_received: list[TypeBEvent] = []

        bus.subscribe(TypeAEvent, lambda e: type_a_received.append(e))
        bus.subscribe(TypeBEvent, lambda e: type_b_received.append(e))

        for v in type_a_values:
            bus.emit(TypeAEvent(value=v))
        for v in type_b_values:
            bus.emit(TypeBEvent(value=v))

        assert len(type_a_received) == len(type_a_values)
        assert len(type_b_received) == len(type_b_values)
