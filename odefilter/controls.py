"""Control algorithms."""

import abc
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class AbstractControl(abc.ABC):
    """Interface for control algorithms."""

    @abc.abstractmethod
    def init_fn(self):
        """Initialise a controller state."""
        raise NotImplementedError

    @abc.abstractmethod
    def control_fn(self, *, state, error_normalised, error_order):
        """Control a normalised error estimate."""
        raise NotImplementedError


class PIState(NamedTuple):
    """Proportional-integral controller state."""

    scale_factor: float
    error_norm_previously_accepted: float


@register_pytree_node_class
@dataclass(frozen=True)
class ProportionalIntegral(AbstractControl):
    """PI Controller."""

    safety: float = 0.95
    factor_min: float = 0.2
    factor_max: float = 10.0
    power_integral_unscaled: float = 0.3
    power_proportional_unscaled: float = 0.4

    def tree_flatten(self):
        children = (
            self.safety,
            self.factor_min,
            self.factor_max,
            self.power_integral_unscaled,
            self.power_proportional_unscaled,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    def init_fn(self):
        """Initialise a controller state."""
        return PIState(scale_factor=1.0, error_norm_previously_accepted=1.0)

    def control_fn(self, *, state, error_normalised, error_order):
        """Control a normalised error estimate."""
        scale_factor = self._scale_factor_proportional_integral(
            error_norm=error_normalised,
            error_order=error_order,
            error_norm_previously_accepted=state.error_norm_previously_accepted,
            safety=self.safety,
            factor_min=self.factor_min,
            factor_max=self.factor_max,
            power_integral_unscaled=self.power_integral_unscaled,
            power_proportional_unscaled=self.power_proportional_unscaled,
        )
        error_norm_previously_accepted = jnp.where(
            error_normalised <= 1.0,
            error_normalised,
            state.error_norm_previously_accepted,
        )
        return PIState(
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


class IState(NamedTuple):
    """Integral controller state."""

    scale_factor: float


class Integral(AbstractControl):
    """Integral control."""

    safety: float = 0.95
    factor_min: float = 0.2
    factor_max: float = 10.0

    def init_fn(self):
        """Initialise a controller state."""
        return IState(scale_factor=1.0)

    def control_fn(self, state, error_normalised, error_order):
        """Control a normalised error estimate."""
        scale_factor = self._scale_factor_integral_control(
            error_norm=error_normalised,
            error_order=error_order,
            safety=self.safety,
            factor_min=self.factor_min,
            factor_max=self.factor_max,
        )
        return IState(scale_factor=scale_factor)

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
