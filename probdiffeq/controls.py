"""Step-size control algorithms."""

import dataclasses
import functools
from typing import Callable, Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers

S = TypeVar("S")
"""Controller state."""


@dataclasses.dataclass
class Control(Generic[S]):
    """Control algorithm."""

    init: Callable
    """Initialise the controller state."""

    clip: Callable
    """(Optionally) clip the current step to not exceed t1."""

    apply: Callable
    r"""Propose a time-step $\Delta t$."""

    extract: Callable
    """Extract the time-step from the controller state."""


class _PIState(containers.NamedTuple):
    """Proportional-integral controller state."""

    dt_proposed: float
    error_norm_previously_accepted: float


def proportional_integral(**options) -> Control[_PIState]:
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    return Control(init=init, apply=apply, extract=extract, clip=_no_clip)


def proportional_integral_clipped(**options) -> Control[_PIState]:
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    clip = _proportional_integral_clip
    return Control(init=init, apply=apply, extract=extract, clip=clip)


def _proportional_integral_init(dt0, /):
    return _PIState(dt_proposed=dt0, error_norm_previously_accepted=1.0)


def _proportional_integral_clip(state: _PIState, /, t, *, t1) -> _PIState:
    dt = state.dt_proposed
    dt_clipped = jnp.minimum(dt, t1 - t)
    return _PIState(dt_clipped, state.error_norm_previously_accepted)


def _proportional_integral_apply(
    state: _PIState,
    /,
    error_normalised,
    error_contraction_rate,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _PIState:
    n1 = power_integral_unscaled / error_contraction_rate
    n2 = power_proportional_unscaled / error_contraction_rate

    a1 = (1.0 / error_normalised) ** n1
    a2 = (state.error_norm_previously_accepted / error_normalised) ** n2
    scale_factor_unclipped = safety * a1 * a2

    scale_factor_clipped_min = jnp.minimum(scale_factor_unclipped, factor_max)
    scale_factor = jnp.maximum(factor_min, scale_factor_clipped_min)
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


def _proportional_integral_extract(state: _PIState, /):
    return state.dt_proposed


def integral(**options) -> Control[float]:
    init = functools.partial(_integral_init, **options)
    apply = functools.partial(_integral_apply, **options)
    extract = functools.partial(_integral_extract, **options)
    return Control(init=init, apply=apply, extract=extract, clip=_no_clip)


def integral_clipped(**options) -> Control[float]:
    init = functools.partial(_integral_init)
    apply = functools.partial(_integral_apply, **options)
    extract = functools.partial(_integral_extract)
    return Control(init=init, apply=apply, extract=extract, clip=_integral_clip)


def _integral_init(dt0, /):
    return dt0


def _integral_clip(dt, /, t, *, t1):
    return jnp.minimum(dt, t1 - t)


def _no_clip(dt, /, t, *, t1):
    return dt


def _integral_apply(
    dt,
    /,
    error_normalised,
    *,
    error_contraction_rate,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
):
    error_power = error_normalised ** (-1.0 / error_contraction_rate)
    scale_factor_unclipped = safety * error_power

    scale_factor_clipped_min = jnp.minimum(scale_factor_unclipped, factor_max)
    scale_factor = jnp.maximum(factor_min, scale_factor_clipped_min)
    return scale_factor * dt


def _integral_extract(dt, /):
    return dt


# Register the control algorithm as a pytree (temporary?)


def _flatten(ctrl):
    aux = ctrl.init, ctrl.apply, ctrl.clip, ctrl.extract
    return (), aux


def _unflatten(aux, _children):
    init, apply, clip, extract = aux
    return Control(init=init, apply=apply, clip=clip, extract=extract)


jax.tree_util.register_pytree_node(Control, _flatten, _unflatten)

#
# class _ProportionalIntegralCommon(Control[_PIState]):
#     def __init__(
#         self,
#         safety: float = 0.95,
#         factor_min: float = 0.2,
#         factor_max: float = 10.0,
#         power_integral_unscaled: float = 0.3,
#         power_proportional_unscaled: float = 0.4,
#     ):
#         self.safety = safety
#         self.factor_min = factor_min
#         self.factor_max = factor_max
#         self.power_integral_unscaled = power_integral_unscaled
#         self.power_proportional_unscaled = power_proportional_unscaled
#
#     def init(self, dt0):
#         return _PIState(dt_proposed=dt0, error_norm_previously_accepted=1.0)
#
#     def clip(self, state: _PIState, /, t, t1) -> _PIState:
#         raise NotImplementedError
#
#     def apply(
#         self, state: _PIState, /, error_normalised, error_contraction_rate
#     ) -> _PIState:
#         n1 = self.power_integral_unscaled / error_contraction_rate
#         n2 = self.power_proportional_unscaled / error_contraction_rate
#
#         a1 = (1.0 / error_normalised) ** n1
#         a2 = (state.error_norm_previously_accepted / error_normalised) ** n2
#         scale_factor_unclipped = self.safety * a1 * a2
#
#         scale_factor_clipped_min = jnp.minimum(scale_factor_unclipped, self.factor_max)
#         scale_factor = jnp.maximum(self.factor_min, scale_factor_clipped_min)
#         error_norm_previously_accepted = jnp.where(
#             error_normalised <= 1.0,
#             error_normalised,
#             state.error_norm_previously_accepted,
#         )
#
#         dt_proposed = scale_factor * state.dt_proposed
#         state = _PIState(
#             dt_proposed=dt_proposed,
#             error_norm_previously_accepted=error_norm_previously_accepted,
#         )
#         return state
#
#     def extract(self, state: _PIState, /) -> float:
#         return state.dt_proposed
#
#
# class ProportionalIntegral(_ProportionalIntegralCommon):
#     """Proportional-integral (PI) controller."""
#
#     def clip(self, state: _PIState, /, t, t1) -> _PIState:
#         return state
#
#
# class ProportionalIntegralClipped(_ProportionalIntegralCommon):
#     r"""Proportional-integral (PI) controller.
#
#     Suggested time-steps are always clipped to $\min(\Delta t, t_1-t)$.
#     """
#
#     def clip(self, state: _PIState, /, t, t1) -> _PIState:
#         dt = state.dt_proposed
#         dt_clipped = jnp.minimum(dt, t1 - t)
#         return _PIState(dt_clipped, state.error_norm_previously_accepted)
#
#
# class _IntegralCommon(Control[float]):
#     def __init__(
#         self, safety: float = 0.95, factor_min: float = 0.2, factor_max: float = 10.0
#     ):
#         self.safety = safety
#         self.factor_min = factor_min
#         self.factor_max = factor_max
#
#     def init(self, dt0) -> float:
#         return dt0
#
#     def clip(self, dt: float, /, t, t1) -> float:
#         raise NotImplementedError
#
#     def apply(self, dt, /, error_normalised, error_contraction_rate) -> float:
#         error_power = error_normalised ** (-1.0 / error_contraction_rate)
#         scale_factor_unclipped = self.safety * error_power
#
#         scale_factor_clipped_min = jnp.minimum(scale_factor_unclipped, self.factor_max)
#         scale_factor = jnp.maximum(self.factor_min, scale_factor_clipped_min)
#         return scale_factor * dt
#
#     def extract(self, dt: float, /) -> float:
#         return dt
#
#
# class Integral(_IntegralCommon):
#     r"""Integral (I) controller."""
#
#     def clip(self, dt: float, /, t, t1) -> float:
#         return dt
#
#
# class IntegralClipped(_IntegralCommon):
#     r"""Integral (I) controller.
#
#     Time-steps are always clipped to $\min(\Delta t, t_1-t)$.
#     """
#
#     def clip(self, dt: float, /, t, t1) -> float:
#         dt_clipped = jnp.minimum(dt, t1 - t)
#         return dt_clipped
