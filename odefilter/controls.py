"""Step-size selection."""

import abc
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AbstractControl(abc.ABC):
    """Interface for control algorithms."""

    @abc.abstractmethod
    def init_fn(self):
        raise NotImplementedError

    @abc.abstractmethod
    def control_fn(self, state, error_normalised, error_order, params):
        raise NotImplementedError


def proportional_integral(**kwargs):
    """Proportional-integral control."""
    return _PIControl(), _PIControl.Params(**kwargs)


class _PIControl(AbstractControl):
    class Params(NamedTuple):
        safety: float = 0.95
        factor_min: float = 0.2
        factor_max: float = 10.0
        power_integral_unscaled: float = 0.3
        power_proportional_unscaled: float = 0.4

    class State(NamedTuple):
        scale_factor: float
        error_norm_previously_accepted: float

    def init_fn(self):
        return self.State(scale_factor=1.0, error_norm_previously_accepted=1.0)

    def control_fn(self, *, state, error_normalised, error_order, params):
        scale_factor = self._scale_factor_proportional_integral(
            error_norm=error_normalised,
            error_order=error_order,
            error_norm_previously_accepted=state.error_norm_previously_accepted,
            safety=params.safety,
            factor_min=params.factor_min,
            factor_max=params.factor_max,
            power_integral_unscaled=params.power_integral_unscaled,
            power_proportional_unscaled=params.power_proportional_unscaled,
        )
        error_norm_previously_accepted = jnp.where(
            error_normalised <= 1.0,
            error_normalised,
            state.error_norm_previously_accepted,
        )
        return state._replace(
            scale_factor=scale_factor,
            error_norm_previously_accepted=error_norm_previously_accepted,
        )

    @staticmethod
    def _scale_factor_proportional_integral(
        *,
        error_norm,
        error_norm_previously_accepted,
        error_order,
        safety,
        factor_min,
        factor_max,
        power_integral_unscaled,
        power_proportional_unscaled,
    ):
        """Proportional-integral control.

        Proportional-integral control simplifies to integral control
        when the parameters are chosen as

            `power_integral_unscaled=1`,
            `power_proportional_unscaled=0`.
        """
        n1 = power_integral_unscaled / error_order
        n2 = power_proportional_unscaled / error_order

        a1 = (1.0 / error_norm) ** n1
        a2 = (error_norm_previously_accepted / error_norm) ** n2
        scale_factor = safety * a1 * a2

        scale_factor_clipped = jnp.maximum(
            factor_min, jnp.minimum(scale_factor, factor_max)
        )
        return scale_factor_clipped


def integral(**kwargs):
    """Integral control."""
    return _IControl(), _IControl.Params(**kwargs)


class _IControl(AbstractControl):
    class Params(NamedTuple):
        safety: float = 0.95
        factor_min: float = 0.2
        factor_max: float = 10.0

    class State(NamedTuple):
        scale_factor: float

    def init_fn(self):
        return self.State(scale_factor=1.0)

    def control_fn(self, state, error_normalised, error_order, params):
        scale_factor = self._scale_factor_integral_control(
            error_norm=error_normalised,
            error_order=error_order,
            safety=params.safety,
            factor_min=params.factor_min,
            factor_max=params.factor_max,
        )
        return self.State(scale_factor=scale_factor)

    @staticmethod
    def _scale_factor_integral_control(
        *, error_norm, safety, error_order, factor_min, factor_max
    ):
        """Integral control."""
        scale_factor = safety * (error_norm ** (-1.0 / error_order))
        scale_factor_clipped = jnp.maximum(
            factor_min, jnp.minimum(scale_factor, factor_max)
        )
        return scale_factor_clipped
