"""Step-size control algorithms."""

import abc
import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AbstractControl(abc.ABC):
    @abc.abstractmethod
    def init_fn(self):
        """Initialise the controller state."""
        raise NotImplementedError

    @abc.abstractmethod
    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous
    ):
        r"""Propose a time-step $\Delta t$."""
        raise NotImplementedError

    @abc.abstractmethod
    def clip_fn(self, *, state, dt, t1):
        raise NotImplementedError


class _PIState(NamedTuple):
    """Proportional-integral controller state."""

    error_norm_previously_accepted: float


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _ProportionalIntegralCommon(AbstractControl):
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

    @abc.abstractmethod
    def clip_fn(self, *, state, dt, t1):
        raise NotImplementedError

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous
    ):
        n1 = self.power_integral_unscaled / error_contraction_rate
        n2 = self.power_proportional_unscaled / error_contraction_rate

        a1 = (1.0 / error_normalised) ** n1
        a2 = (state.error_norm_previously_accepted / error_normalised) ** n2
        scale_factor_unclipped = self.safety * a1 * a2

        scale_factor = jnp.maximum(
            self.factor_min, jnp.minimum(scale_factor_unclipped, self.factor_max)
        )
        error_norm_previously_accepted = jnp.where(
            error_normalised <= 1.0,
            error_normalised,
            state.error_norm_previously_accepted,
        )
        state = _PIState(error_norm_previously_accepted=error_norm_previously_accepted)
        return scale_factor * dt_previous, state


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ProportionalIntegral(_ProportionalIntegralCommon):
    """Proportional-integral (PI) controller."""

    def clip_fn(self, *, state, dt, t1):
        return dt


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ProportionalIntegralClipped(_ProportionalIntegralCommon):
    r"""Proportional-integral (PI) controller.

    Suggested time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip_fn(self, *, state, dt, t1):
        return jnp.minimum(dt, t1 - state.t)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
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

    @abc.abstractmethod
    def clip_fn(self, *, state, dt, t1):
        raise NotImplementedError

    def control_fn(
        self, *, state, error_normalised, error_contraction_rate, dt_previous
    ):
        scale_factor_unclipped = self.safety * (
            error_normalised ** (-1.0 / error_contraction_rate)
        )

        scale_factor = jnp.maximum(
            self.factor_min, jnp.minimum(scale_factor_unclipped, self.factor_max)
        )
        return scale_factor * dt_previous, ()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Integral(_IntegralCommon):
    r"""Integral (I) controller."""

    def clip_fn(self, *, state, dt, t1):
        return dt


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IntegralClipped(_IntegralCommon):
    r"""Integral (I) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip_fn(self, *, state, dt, t1):
        return jnp.minimum(dt, t1 - state.t)
