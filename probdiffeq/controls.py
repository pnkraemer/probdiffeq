"""Step-size control algorithms."""

import functools
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers

S = TypeVar("S")
"""Controller state."""


class Control(Generic[S]):
    """Interface for control-algorithms."""

    def init(self, dt0: float) -> S:
        """Initialise the controller state."""
        raise NotImplementedError

    def clip(self, state: S, /, t: float, t1: float) -> S:
        """(Optionally) clip the current step to not exceed t1."""
        raise NotImplementedError

    def apply(
        self, state, /, error_normalised: float, error_contraction_rate: float
    ) -> S:
        r"""Propose a time-step $\Delta t$."""
        raise NotImplementedError

    def extract(self, state: S, /) -> float:
        """Extract the time-step from the controller state."""
        raise NotImplementedError


class _PIState(containers.NamedTuple):
    """Proportional-integral controller state."""

    dt_proposed: float
    error_norm_previously_accepted: float


class _ProportionalIntegralCommon(Control[_PIState]):
    def __init__(
        self,
        *,
        safety: float = 0.95,
        factor_min: float = 0.2,
        factor_max: float = 10.0,
        power_integral_unscaled: float = 0.3,
        power_proportional_unscaled: float = 0.4,
    ):
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.power_integral_unscaled = power_integral_unscaled
        self.power_proportional_unscaled = power_proportional_unscaled

    def init(self, dt0):
        return _PIState(dt_proposed=dt0, error_norm_previously_accepted=1.0)

    def clip(self, state: _PIState, /, t, t1) -> _PIState:
        raise NotImplementedError

    def apply(
        self, state: _PIState, /, error_normalised, error_contraction_rate
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

    def extract(self, state: _PIState, /) -> float:
        return state.dt_proposed


class ProportionalIntegral(_ProportionalIntegralCommon):
    """Proportional-integral (PI) controller."""

    def clip(self, state: _PIState, /, t, t1) -> _PIState:
        return state


class ProportionalIntegralClipped(_ProportionalIntegralCommon):
    r"""Proportional-integral (PI) controller.

    Suggested time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip(self, state: _PIState, /, t, t1) -> _PIState:
        dt = state.dt_proposed
        dt_clipped = jnp.minimum(dt, t1 - t)
        return _PIState(dt_clipped, state.error_norm_previously_accepted)


class _IState(containers.NamedTuple):
    dt_proposed: jax.Array


class _IntegralCommon(Control[_IState]):
    def __init__(
        self, *, safety: float = 0.95, factor_min: float = 0.2, factor_max: float = 10.0
    ):
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max

    def init(self, dt0) -> _IState:
        return _IState(dt0)

    def clip(self, state: _IState, /, t, t1) -> _IState:
        raise NotImplementedError

    def apply(
        self, state: _IState, /, error_normalised, error_contraction_rate
    ) -> _IState:
        scale_factor_unclipped = self.safety * (
            error_normalised ** (-1.0 / error_contraction_rate)
        )

        scale_factor = jnp.maximum(
            self.factor_min, jnp.minimum(scale_factor_unclipped, self.factor_max)
        )
        dt = scale_factor * state.dt_proposed
        return _IState(dt)

    def extract(self, state: _IState, /) -> float:
        return state.dt_proposed


class Integral(_IntegralCommon):
    r"""Integral (I) controller."""

    def clip(self, state: _IState, /, t: float, t1: float) -> _IState:
        return state


class IntegralClipped(_IntegralCommon):
    r"""Integral (I) controller.

    Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
    """

    def clip(self, state: _IState, /, t: float, t1: float) -> _IState:
        dt_clipped = jnp.minimum(state.dt_proposed, t1 - t)
        return _IState(dt_clipped)


# Register the controllers as PyTrees
# (we do this outside of the classes to de-clutter the class-code a bit)


def _pi_flatten(pi_controller: _ProportionalIntegralCommon, /):
    children = (
        pi_controller.safety,
        pi_controller.factor_min,
        pi_controller.factor_max,
        pi_controller.power_integral_unscaled,
        pi_controller.power_proportional_unscaled,
    )
    aux = ()
    return children, aux


def _i_flatten(i_controller: _IntegralCommon, /):
    children = (i_controller.safety, i_controller.factor_min, i_controller.factor_max)
    aux = ()
    return children, aux


def _unflatten(_aux, children, *, clz):
    return clz(*children)


for x in [ProportionalIntegral, ProportionalIntegralClipped]:
    jax.tree_util.register_pytree_node(
        x, _pi_flatten, functools.partial(_unflatten, clz=x)
    )

for y in [Integral, IntegralClipped]:
    jax.tree_util.register_pytree_node(
        y, _i_flatten, functools.partial(_unflatten, clz=y)
    )
