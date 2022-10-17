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
        """Initialise the controller state."""
        raise NotImplementedError

    # todo: rename error_contraction_rate to error_contraction_rate
    @abc.abstractmethod
    def control_fn(self, *, state, error_normalised, error_order, dt_previous, t, t1):
        r"""Propose a time-step $\Delta t$.

        A good time-step $\Delta t$ is as large as possible such that the normalised error
        is smaller than 1. This is commonly a function of previous error estimates,
        the current normalised error, and some expected local convergence rate / error order.
        """
        raise NotImplementedError


class _PIState(NamedTuple):
    """Proportional-integral controller state."""

    error_norm_previously_accepted: float


@register_pytree_node_class
@dataclass(frozen=True)
class _ProportionalIntegralCommon(AbstractControl):
    """Proportional-integral (PI) controller."""

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
        return _PIState(error_norm_previously_accepted=1.0)

    @staticmethod
    def _scale_factor_proportional_integral(
        *,
        error_norm,
        error_norm_previously_accepted,
        error_contraction_rate,
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
        n1 = power_integral_unscaled / error_contraction_rate
        n2 = power_proportional_unscaled / error_contraction_rate

        a1 = (1.0 / error_norm) ** n1
        a2 = (error_norm_previously_accepted / error_norm) ** n2
        scale_factor = safety * a1 * a2

        scale_factor_clipped = jnp.maximum(
            factor_min, jnp.minimum(scale_factor, factor_max)
        )
        return scale_factor_clipped

    @abc.abstractmethod
    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):
        raise NotImplementedError


@register_pytree_node_class
@dataclass(frozen=True)
class ProportionalIntegral(_ProportionalIntegralCommon):
    """Proportional-integral (PI) controller."""

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):

        scale_factor = self._scale_factor_proportional_integral(
            error_norm=error_normalised,
            error_contraction_rate=error_contraction_rate,
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
        state = _PIState(error_norm_previously_accepted=error_norm_previously_accepted)
        return scale_factor * dt_previous, state


@register_pytree_node_class
@dataclass(frozen=True)
class ClippedProportionalIntegral(_ProportionalIntegralCommon):
    r"""Proportional-integral (PI) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):
        scale_factor = self._scale_factor_proportional_integral(
            error_norm=error_normalised,
            error_contraction_rate=error_contraction_rate,
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
        state = _PIState(error_norm_previously_accepted=error_norm_previously_accepted)
        dt_proposed = jnp.minimum(scale_factor * dt_previous, t1 - t)
        return dt_proposed, state


@register_pytree_node_class
@dataclass(frozen=True)
class _IntegralCommon(AbstractControl):

    safety: float = 0.95
    factor_min: float = 0.2
    factor_max: float = 10.0

    def tree_flatten(self):
        children = (self.safety, self.factor_min, self.factor_max)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    def init_fn(self):
        return ()

    @staticmethod
    def _scale_factor_integral_control(
        *, error_norm, safety, error_contraction_rate, factor_min, factor_max
    ):
        scale_factor = safety * (error_norm ** (-1.0 / error_contraction_rate))
        scale_factor_clipped = jnp.maximum(
            factor_min, jnp.minimum(scale_factor, factor_max)
        )
        return scale_factor_clipped

    @abc.abstractmethod
    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):
        raise NotImplementedError


@register_pytree_node_class
@dataclass(frozen=True)
class Integral(_IntegralCommon):
    r"""Integral (I) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):
        scale_factor = self._scale_factor_integral_control(
            error_norm=error_normalised,
            error_contraction_rate=error_contraction_rate,
            safety=self.safety,
            factor_min=self.factor_min,
            factor_max=self.factor_max,
        )
        return scale_factor * dt_previous, ()


@register_pytree_node_class
@dataclass(frozen=True)
class ClippedIntegral(_IntegralCommon):
    r"""Integral (I) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous, t, t1
    ):
        scale_factor = self._scale_factor_integral_control(
            error_norm=error_normalised,
            error_contraction_rate=error_contraction_rate,
            safety=self.safety,
            factor_min=self.factor_min,
            factor_max=self.factor_max,
        )
        dt_proposed = jnp.minimum(scale_factor * dt_previous, t1 - t)
        return dt_proposed, ()
