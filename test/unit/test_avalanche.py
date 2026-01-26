"""Unit tests for avalanche detection.

RED phase: These tests should fail until AvalancheDetector is implemented.

An avalanche is a burst of neural activity:
1. Starts when firing count rises above quiet threshold
2. Ends when firing count returns below quiet threshold
3. Size = sum of all firing events during avalanche
4. Duration = number of time steps
"""

import numpy as np
import pytest

from src.events.avalanche import AvalancheDetector, Avalanche


class TestAvalancheDataStructure:
    """Tests for Avalanche data structure."""

    def test_avalanche_has_size(self):
        """Avalanche should have size attribute."""
        avalanche = Avalanche(size=100, duration=5, peak=30, start_time=10)
        assert avalanche.size == 100

    def test_avalanche_has_duration(self):
        """Avalanche should have duration attribute."""
        avalanche = Avalanche(size=100, duration=5, peak=30, start_time=10)
        assert avalanche.duration == 5

    def test_avalanche_has_peak(self):
        """Avalanche should have peak firing count."""
        avalanche = Avalanche(size=100, duration=5, peak=30, start_time=10)
        assert avalanche.peak == 30

    def test_avalanche_has_start_time(self):
        """Avalanche should record when it started."""
        avalanche = Avalanche(size=100, duration=5, peak=30, start_time=10)
        assert avalanche.start_time == 10


class TestAvalancheDetectorCreation:
    """Tests for AvalancheDetector initialization."""

    def test_detector_accepts_quiet_threshold(self):
        """Detector should accept quiet threshold as fraction."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.05)
        assert detector.quiet_threshold == 0.05
        assert detector.quiet_count == 5  # 5% of 100

    def test_detector_accepts_n_neurons(self):
        """Detector should know network size."""
        detector = AvalancheDetector(n_neurons=300, quiet_threshold=0.1)
        assert detector.n_neurons == 300

    def test_detector_starts_with_no_avalanches(self):
        """Detector should start with empty avalanche list."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.05)
        assert len(detector.avalanches) == 0


class TestAvalancheDetection:
    """Tests for avalanche detection logic."""

    def test_no_avalanche_when_always_quiet(self):
        """No avalanche should be detected if firing stays below threshold."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1)
        
        # All firing counts below 10 (10% of 100)
        for t in range(10):
            detector.record_step(time_step=t, firing_count=5)
        
        assert len(detector.avalanches) == 0

    def test_detects_simple_avalanche(self):
        """Detect an avalanche that rises above and returns below threshold."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        # Quiet -> active -> quiet
        detector.record_step(time_step=0, firing_count=5)   # quiet
        detector.record_step(time_step=1, firing_count=15)  # active
        detector.record_step(time_step=2, firing_count=20)  # active
        detector.record_step(time_step=3, firing_count=12)  # active
        detector.record_step(time_step=4, firing_count=5)   # quiet - avalanche ends
        
        assert len(detector.avalanches) == 1
        avalanche = detector.avalanches[0]
        assert avalanche.size == 15 + 20 + 12  # 47
        assert avalanche.duration == 3
        assert avalanche.peak == 20
        assert avalanche.start_time == 1

    def test_detects_multiple_avalanches(self):
        """Detect multiple separate avalanches."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        # First avalanche
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=15)
        detector.record_step(time_step=2, firing_count=5)
        
        # Quiet period
        detector.record_step(time_step=3, firing_count=3)
        
        # Second avalanche
        detector.record_step(time_step=4, firing_count=25)
        detector.record_step(time_step=5, firing_count=30)
        detector.record_step(time_step=6, firing_count=5)
        
        assert len(detector.avalanches) == 2
        assert detector.avalanches[0].size == 15
        assert detector.avalanches[1].size == 55

    def test_avalanche_at_exact_threshold(self):
        """Activity exactly at threshold should be considered active."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=10)  # exactly at 10%
        detector.record_step(time_step=2, firing_count=5)
        
        assert len(detector.avalanches) == 1


class TestAvalancheMetrics:
    """Tests for avalanche statistics."""

    def test_branching_ratio_single_step(self):
        """Branching ratio = avg(firing_t+1 / firing_t)."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        # During avalanche: 10 -> 20 (ratio 2.0) -> 10 (ratio 0.5)
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=10)
        detector.record_step(time_step=2, firing_count=20)
        detector.record_step(time_step=3, firing_count=10)
        detector.record_step(time_step=4, firing_count=5)
        
        ratio = detector.compute_branching_ratio()
        # (20/10 + 10/20) / 2 = (2.0 + 0.5) / 2 = 1.25
        assert ratio == pytest.approx(1.25, rel=0.01)

    def test_branching_ratio_critical(self):
        """Branching ratio ≈ 1.0 indicates critical dynamics."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        # Constant activity during avalanche
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=15)
        detector.record_step(time_step=2, firing_count=15)
        detector.record_step(time_step=3, firing_count=15)
        detector.record_step(time_step=4, firing_count=5)
        
        ratio = detector.compute_branching_ratio()
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_size_distribution(self):
        """Get distribution of avalanche sizes."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        # Create several avalanches of different sizes
        for i in range(3):
            detector.record_step(time_step=i*3, firing_count=5)
            detector.record_step(time_step=i*3+1, firing_count=10 + i*10)
            detector.record_step(time_step=i*3+2, firing_count=5)
        
        sizes = detector.get_size_distribution()
        assert len(sizes) == 3
        assert sizes == [10, 20, 30]


class TestOngoingAvalanche:
    """Tests for handling ongoing (unclosed) avalanches."""

    def test_ongoing_avalanche_not_counted(self):
        """Avalanche still in progress should not be in completed list."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=15)  # starts
        detector.record_step(time_step=2, firing_count=20)  # ongoing
        # Not yet returned to quiet
        
        assert len(detector.avalanches) == 0  # not closed yet
        assert detector.is_in_avalanche

    def test_finalize_closes_ongoing_avalanche(self):
        """Finalize should close any ongoing avalanche."""
        detector = AvalancheDetector(n_neurons=100, quiet_threshold=0.1, burn_in=0)
        
        detector.record_step(time_step=0, firing_count=5)
        detector.record_step(time_step=1, firing_count=15)
        detector.record_step(time_step=2, firing_count=20)
        
        detector.finalize()
        
        assert len(detector.avalanches) == 1
        assert detector.avalanches[0].size == 35
