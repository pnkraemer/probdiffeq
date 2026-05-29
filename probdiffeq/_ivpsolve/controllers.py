from probdiffeq.backend import np
from probdiffeq.backend.typing import Generic, TypeVar

T = TypeVar("T")
S = TypeVar("S")


__all__ = ["Control", "control_integral", "control_proportional_integral"]


class Control(Generic[T]):
    """An interface for a control algorithm."""

    def init(self, dt: float, /) -> T:
        """Initialise the controller state."""
        raise NotImplementedError

    def apply(self, dt: float, state: T, /, *, error_power: float) -> tuple[float, T]:
        """Propose a time-step-size."""
        raise NotImplementedError


# TODO: these parameters are floats and could be passed to "apply"?
class control_proportional_integral(Control[float]):
    """Construct a proportional-integral-controller with time-clipping."""

    def __init__(
        self,
        *,
        safety=0.95,
        factor_min=0.2,
        factor_max=10.0,
        power_integral_unscaled=0.3,
        power_proportional_unscaled=0.4,
    ) -> None:
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.power_integral_unscaled = power_integral_unscaled
        self.power_proportional_unscaled = power_proportional_unscaled

    def init(self, dt: float, /) -> float:
        del dt
        return 1.0

    def apply(self, dt: float, error_power_prev: float, /, *, error_power):
        # Equivalent: error_power = error_norm ** (-1.0 / error_contraction_rate)
        a1 = error_power**self.power_integral_unscaled
        a2 = (error_power / error_power_prev) ** self.power_proportional_unscaled
        scale_factor_unclipped = self.safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, self.factor_max)
        scale_factor = np.maximum(self.factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt
        return dt_proposed, error_power_prev


class control_integral(Control[tuple]):
    """Construct an integral-controller."""

    def __init__(self, *, safety=0.95, factor_min=0.2, factor_max=10.0) -> None:
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max

    def init(self, dt, /) -> tuple:
        del dt
        return ()

    def apply(self, dt, state: tuple, /, *, error_power):
        del state
        scale_factor_unclipped = self.safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, self.factor_max)
        scale_factor = np.maximum(self.factor_min, scale_factor_clipped_min)
        return scale_factor * dt, ()
