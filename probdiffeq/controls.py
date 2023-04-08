"""Step-size control algorithms."""

import abc
import dataclasses
from typing import Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp

S = TypeVar("S")
"""Controller state."""


class AbstractControl(abc.ABC, Generic[S]):
    """Interface for control-algorithms."""

    @abc.abstractmethod
    def init_state_from_dt(self, dt0: jax.Array) -> S:
        """Initialise the controller state."""
        raise NotImplementedError

    @abc.abstractmethod
    def clip(self, t: jax.Array, t1: jax.Array, state: S) -> S:
        """(Optionally) clip the current step to not exceed t1."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self, error_normalised: jax.Array, error_contraction_rate: jax.Array, state: S
    ) -> S:
        r"""Propose a time-step $\Delta t$."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_dt_from_state(self, state: S) -> jax.Array:
        """Extract the time-step from the controller state."""
        raise NotImplementedError


class _PIState(NamedTuple):
    """Proportional-integral controller state."""

    dt_proposed: jax.Array  # usually a float
    error_norm_previously_accepted: jax.Array  # usually a float


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _ProportionalIntegralCommon(AbstractControl[_PIState]):
    safety: jax.Array = 0.95
    factor_min: jax.Array = 0.2
    factor_max: jax.Array = 10.0
    power_integral_unscaled: jax.Array = 0.3
    power_proportional_unscaled: jax.Array = 0.4

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

    def init_state_from_dt(self, dt0):
        return _PIState(dt_proposed=dt0, error_norm_previously_accepted=1.0)

    @abc.abstractmethod
    def clip(self, t, t1, state: _PIState) -> _PIState:
        raise NotImplementedError

    def apply(
        self, error_normalised, error_contraction_rate, state: _PIState
    ) -> _PIState:
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

        dt_proposed = scale_factor * state.dt_proposed
        state = _PIState(
            dt_proposed=dt_proposed,
            error_norm_previously_accepted=error_norm_previously_accepted,
        )
        return state

    def extract_dt_from_state(self, state: _PIState) -> jax.Array:
        return state.dt_proposed


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ProportionalIntegral(_ProportionalIntegralCommon):
    """Proportional-integral (PI) controller."""

    def clip(self, t, t1, state: _PIState) -> _PIState:
        return state


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ProportionalIntegralClipped(_ProportionalIntegralCommon):
    r"""Proportional-integral (PI) controller.

    Suggested time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip(self, t, t1, state: _PIState) -> _PIState:
        dt = state.dt_proposed
        dt_clipped = jnp.minimum(dt, t1 - t)
        return _PIState(dt_clipped, state.error_norm_previously_accepted)


class _IState(NamedTuple):
    dt_proposed: jax.Array  # usually a float


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _IntegralCommon(AbstractControl[_IState]):
    safety: jax.Array = 0.95
    factor_min: jax.Array = 0.2
    factor_max: jax.Array = 10.0

    def tree_flatten(self):
        children = (self.safety, self.factor_min, self.factor_max)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    def init_state_from_dt(self, dt0) -> _IState:
        return _IState(dt0)

    @abc.abstractmethod
    def clip(self, t, t1, state: _IState) -> _IState:
        raise NotImplementedError

    def apply(
        self, error_normalised, error_contraction_rate, state: _IState
    ) -> _IState:
        scale_factor_unclipped = self.safety * (
            error_normalised ** (-1.0 / error_contraction_rate)
        )

        scale_factor = jnp.maximum(
            self.factor_min, jnp.minimum(scale_factor_unclipped, self.factor_max)
        )
        dt = scale_factor * state.dt_proposed
        return _IState(dt)

    def extract_dt_from_state(self, state: _IState) -> jax.Array:
        return state.dt_proposed


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Integral(_IntegralCommon):
    r"""Integral (I) controller."""

    def clip(self, t: jax.Array, t1: jax.Array, state: _IState) -> _IState:
        return state


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IntegralClipped(_IntegralCommon):
    r"""Integral (I) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip(self, t: jax.Array, t1: jax.Array, state: _IState) -> _IState:
        dt_clipped = jnp.minimum(state.dt_proposed, t1 - t)
        return _IState(dt_clipped)
