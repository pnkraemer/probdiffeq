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
        exponent_integral=0.3,
        exponent_proportional=0.4,
    ) -> None:
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.exponent_integral = exponent_integral
        self.exponent_proportional = exponent_proportional

    def init(self, dt: float, /) -> float:
        del dt
        return 1.0

    def apply(self, dt: float, error_norm_inv_prev: float, /, *, error_power):
        # Equivalent: error_power = error_norm ** (-1.0 / error_contraction_rate)
        gain_integral = error_power**self.exponent_integral
        gain_proportional = (
            error_power / error_norm_inv_prev
        ) ** self.exponent_proportional
        step_ratio_unclipped = self.safety * gain_integral * gain_proportional

        scale_factor_clipped_min = np.minimum(step_ratio_unclipped, self.factor_max)
        scale_factor = np.maximum(self.factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_norm_inv_prev = np.where(
            error_power >= 1.0, error_power, error_norm_inv_prev
        )

        dt_proposed = scale_factor * dt
        return dt_proposed, error_norm_inv_prev


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
        step_ratio_unclipped = self.safety * error_power

        scale_factor_clipped_min = np.minimum(step_ratio_unclipped, self.factor_max)
        scale_factor = np.maximum(self.factor_min, scale_factor_clipped_min)
        return scale_factor * dt, ()
