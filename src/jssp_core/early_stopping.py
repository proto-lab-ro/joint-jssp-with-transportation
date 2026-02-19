"""Early stopping base class and implementations for training."""

from abc import ABC, abstractmethod


class EarlyStoppingBase(ABC):
    """Base class for early stopping strategies.

    This class provides a framework for implementing early stopping during training.
    Child classes must implement state initialization and update logic.

    Args:
        patience: Number of epochs to wait before stopping if no improvement
        mode: Either 'min' or 'max' - whether lower or higher values are better
        target: Target threshold for stopping. Can be:
            - float: Static threshold value
            - str: Dynamic threshold key to look up in state
            - None: No threshold, only use patience
        state: Dictionary to track early stopping state. Must contain 'streak' key.
        gap: Minimum improvement gap to consider as improvement (default: 0.0)
    """

    def __init__(
        self,
        patience: int,
        mode: str,
        target: float | str | None,
        gap: float = 0.0,
    ) -> None:
        """Initialize early stopping base class."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.patience = patience
        self.mode = mode
        self.target = target
        self.state = {}
        self.gap = gap

        # Initialize state to 0 streak
        self.state["streak"] = 0

        # Allow child classes to initialize additional state
        self._initialize_state()

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize state variables specific to the child class.

        This method is called during __init__ and should set up any additional
        state variables needed by the specific early stopping implementation.
        """
        pass

    @abstractmethod
    def _update_state(self, current_value: float) -> None:
        """Update state variables based on the current value.

        This method is called during check_stop and should update any state
        variables (except streak, which is handled by the base class) based
        on the current metric value.

        Args:
            current_value: The current metric value to evaluate
        """
        pass

    def check_stop(self, current_value: float) -> bool:
        """Check if training should stop based on current value.

        Args:
            current_value: Current metric value to evaluate

        Returns:
            True if training should stop, False otherwise
        """

        # Check target threshold if specified
        self.improvement = self._check_target_threshold(current_value)
        self._update_state(current_value)
        # Check patience-based stopping

        return self.state["streak"] >= self.patience

    def _check_target_threshold(self, current_value: float) -> bool:
        """Check if target threshold has been reached.

        Args:
            current_value: Current metric value

        Returns:
            True if target threshold reached, False otherwise
        """
        if self.target is None:
            return False

        # Get threshold value
        if isinstance(self.target, str):
            # Dynamic threshold from state
            threshold = self.state.get(self.target)
            if threshold is None:
                return False
        else:
            # Static threshold
            threshold = self.target

        # Check if threshold met based on mode
        if self.mode == "min":
            return current_value <= threshold - self.gap
        elif self.mode == "max":  # mode == "max"
            return current_value >= threshold + self.gap
        else:
            raise ValueError(f"Unknown mode '{self.mode}', expected 'min' or 'max'.")

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.state["streak"] = 0
        self._initialize_state()


class NoEarlyStopping(EarlyStoppingBase):
    """Early stopping implementation that never stops training.

    This class is used when no early stopping is desired. It always returns
    False from check_stop, allowing training to continue indefinitely.
    """

    def __init__(self) -> None:
        """Initialize NoEarlyStopping with dummy values."""
        # Use dummy values since they won't be used
        super().__init__(
            patience=1,
            mode="min",
            target=None,
            gap=0.0,
        )

    def _initialize_state(self) -> None:
        """No additional state needed for NoEarlyStopping."""
        pass

    def _update_state(self, current_value: float) -> None:
        """No state updates needed for NoEarlyStopping.

        Args:
            current_value: The current metric value (ignored)
        """
        pass

    def check_stop(self, current_value: float) -> bool:
        """Always return False to never stop training.

        Args:
            current_value: The current metric value (ignored)

        Returns:
            Always False
        """
        return False


def create_early_stopping(config: dict | None = None) -> EarlyStoppingBase:
    """Factory function to create early stopping instances.

    Args:
        config: Configuration dictionary with initialization parameters.
            If None or empty, returns NoEarlyStopping.
            Must contain:
                - name: str - Strategy name ("none", "static", "last_value", "interval", "best_value")
            For StaticEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - target: float
                - gap: float (optional, default 0.0)
            For LastValueEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - gap: float (optional, default 0.0)
            For IntervalEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - gap: float (optional, default 0.0)

    Returns:
        NoEarlyStopping
    """
    # If config is None or empty, return NoEarlyStopping
    if not config:
        return NoEarlyStopping()

    # Extract name from config, default to "none"
    name = config.get("name", "none")
    name_lower = name.lower()

    if name_lower in ("none", "no_early_stopping"):
        return NoEarlyStopping()

    return NoEarlyStopping()
