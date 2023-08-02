"""Step-size control algorithms."""

import dataclasses
import functools
from typing import Any, Callable, Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers

T = TypeVar("T")
"""A type-variable to indicate the controller's state."""


@dataclasses.dataclass
class Controller(Generic[T]):
    """Control algorithm."""

    init: Callable[[Any], T]
    """Initialise the controller state."""

    clip: Callable[[T, Any, Any], T]
    """(Optionally) clip the current step to not exceed t1."""

    apply: Callable[[T, Any, Any], T]
    r"""Propose a time-step $\Delta t$."""

    extract: Callable[[T], Any]
    """Extract the time-step from the controller state."""


class PIState(containers.NamedTuple):
    """Proportional-integral-controller state."""

    dt_proposed: float
    error_norm_previously_accepted: float


def proportional_integral(**options) -> Controller[PIState]:
    """Construct a proportional-integral-controller."""
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    return Controller(init=init, apply=apply, extract=extract, clip=_no_clip)


def proportional_integral_clipped(**options) -> Controller[PIState]:
    """Construct a proportional-integral-controller with time-clipping."""
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    clip = _proportional_integral_clip
    return Controller(init=init, apply=apply, extract=extract, clip=clip)


def _proportional_integral_init(dt0, /):
    return PIState(dt_proposed=dt0, error_norm_previously_accepted=1.0)


def _proportional_integral_clip(state: PIState, /, t, t1) -> PIState:
    dt = state.dt_proposed
    dt_clipped = jnp.minimum(dt, t1 - t)
    return PIState(dt_clipped, state.error_norm_previously_accepted)


def _proportional_integral_apply(
    state: PIState,
    /,
    error_normalised,
    error_contraction_rate,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> PIState:
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
    state = PIState(
        dt_proposed=dt_proposed,
        error_norm_previously_accepted=error_norm_previously_accepted,
    )
    return state


def _proportional_integral_extract(state: PIState, /):
    return state.dt_proposed


def integral(**options) -> Controller[float]:
    """Construct an integral-controller."""
    init = functools.partial(_integral_init, **options)
    apply = functools.partial(_integral_apply, **options)
    extract = functools.partial(_integral_extract, **options)
    return Controller(init=init, apply=apply, extract=extract, clip=_no_clip)


def integral_clipped(**options) -> Controller[float]:
    """Construct an integral-controller with time-clipping."""
    init = functools.partial(_integral_init)
    apply = functools.partial(_integral_apply, **options)
    extract = functools.partial(_integral_extract)
    return Controller(init=init, apply=apply, extract=extract, clip=_integral_clip)


def _integral_init(dt0, /):
    return dt0


def _integral_clip(dt, /, t, t1):
    return jnp.minimum(dt, t1 - t)


def _no_clip(dt, /, t, t1):
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
    return Controller(init=init, apply=apply, clip=clip, extract=extract)


jax.tree_util.register_pytree_node(Controller, _flatten, _unflatten)
