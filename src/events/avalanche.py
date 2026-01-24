"""Avalanche detection for Self-Organized Criticality analysis.

Detects and records avalanches (bursts of neural activity) for SOC analysis.

An avalanche:
1. Starts when firing count rises above quiet threshold
2. Ends when firing count returns below quiet threshold
3. Size = sum of all firing events during avalanche
4. Duration = number of time steps
"""

from dataclasses import dataclass, field


@dataclass
class Avalanche:
    """Record of a completed avalanche.

    Attributes:
        size: Total number of firing events during avalanche
        duration: Number of time steps
        peak: Maximum firing count during avalanche
        start_time: Time step when avalanche started
    """

    size: int
    duration: int
    peak: int
    start_time: int


@dataclass
class AvalancheDetector:
    """Detects and records neural avalanches.

    Attributes:
        n_neurons: Total number of neurons in network
        quiet_threshold: Fraction of neurons below which activity is "quiet"
        burn_in: Number of initial avalanches to skip before recording
        avalanches: List of completed avalanches
    """

    n_neurons: int
    quiet_threshold: float
    burn_in: int = 10

    # Computed from threshold
    quiet_count: int = field(init=False)

    # Completed avalanches
    avalanches: list[Avalanche] = field(default_factory=list, init=False)

    # Count of skipped avalanches during burn-in
    _skipped_count: int = field(default=0, init=False, repr=False)

    # State for tracking ongoing avalanche
    _in_avalanche: bool = field(default=False, init=False, repr=False)
    _current_size: int = field(default=0, init=False, repr=False)
    _current_duration: int = field(default=0, init=False, repr=False)
    _current_peak: int = field(default=0, init=False, repr=False)
    _current_start: int = field(default=0, init=False, repr=False)

    # For branching ratio calculation (accumulated across all avalanches)
    _all_ratios: list[float] = field(default_factory=list, init=False, repr=False)
    _firing_history: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute quiet count from threshold."""
        self.quiet_count = int(self.n_neurons * self.quiet_threshold)

    @property
    def is_in_avalanche(self) -> bool:
        """Return whether an avalanche is currently in progress."""
        return self._in_avalanche

    def record_step(self, time_step: int, firing_count: int) -> None:
        """Record a simulation step for avalanche detection.

        Args:
            time_step: Current simulation time step
            firing_count: Number of neurons firing at this step
        """
        is_active = firing_count >= self.quiet_count

        if not self._in_avalanche:
            # Currently quiet
            if is_active:
                # Avalanche starts
                self._in_avalanche = True
                self._current_size = firing_count
                self._current_duration = 1
                self._current_peak = firing_count
                self._current_start = time_step
                self._firing_history = [firing_count]
        else:
            # Currently in avalanche
            if is_active:
                # Avalanche continues
                self._current_size += firing_count
                self._current_duration += 1
                self._current_peak = max(self._current_peak, firing_count)
                self._firing_history.append(firing_count)
            else:
                # Avalanche ends
                self._close_avalanche()

    def _close_avalanche(self) -> None:
        """Close the current avalanche and add to completed list."""
        # Skip if still in burn-in period
        if self._skipped_count < self.burn_in:
            self._skipped_count += 1
            self._in_avalanche = False
            self._current_size = 0
            self._current_duration = 0
            self._current_peak = 0
            self._firing_history = []
            return

        avalanche = Avalanche(
            size=self._current_size,
            duration=self._current_duration,
            peak=self._current_peak,
            start_time=self._current_start,
        )
        self.avalanches.append(avalanche)

        # Accumulate branching ratios from this avalanche
        for i in range(len(self._firing_history) - 1):
            if self._firing_history[i] > 0:
                self._all_ratios.append(
                    self._firing_history[i + 1] / self._firing_history[i]
                )

        self._in_avalanche = False
        self._current_size = 0
        self._current_duration = 0
        self._current_peak = 0
        self._firing_history = []

    def finalize(self) -> None:
        """Close any ongoing avalanche at end of simulation."""
        if self._in_avalanche:
            self._close_avalanche()

    def compute_branching_ratio(self) -> float:
        """Compute average branching ratio across all avalanches.

        Branching ratio = avg(firing_t+1 / firing_t)
        A ratio of ~1.0 indicates critical dynamics.

        Returns:
            Average branching ratio, or 0.0 if no data
        """
        # Start with accumulated ratios from completed avalanches
        ratios = self._all_ratios.copy()

        # Add ratios from current avalanche if in progress
        if self._in_avalanche:
            for i in range(len(self._firing_history) - 1):
                if self._firing_history[i] > 0:
                    ratios.append(self._firing_history[i + 1] / self._firing_history[i])

        if not ratios:
            return 0.0

        return sum(ratios) / len(ratios)

    def get_size_distribution(self) -> list[int]:
        """Get list of avalanche sizes for distribution analysis.

        Returns:
            List of avalanche sizes in chronological order
        """
        return [a.size for a in self.avalanches]
