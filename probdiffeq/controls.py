"""Step-size control algorithms."""

from probdiffeq.backend import containers, functools, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Callable, Generic, TypeVar

T = TypeVar("T")
"""A type-variable to indicate the controller's state."""


@containers.dataclass
class Controller(Generic[T]):
    """Control algorithm."""

    init: Callable[[float], T]
    """Initialise the controller state."""

    clip: Callable[[T, float, float], T]
    """(Optionally) clip the current step to not exceed t1."""

    apply: Callable[[T, float, float], T]
    r"""Propose a time-step $\Delta t$."""

    extract: Callable[[T], float]
    """Extract the time-step from the controller state."""


def proportional_integral(**options) -> Controller[tuple[float, float]]:
    """Construct a proportional-integral-controller."""
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    return Controller(init=init, apply=apply, extract=extract, clip=_no_clip)


def proportional_integral_clipped(**options) -> Controller[tuple[float, float]]:
    """Construct a proportional-integral-controller with time-clipping."""
    init = _proportional_integral_init
    apply = functools.partial(_proportional_integral_apply, **options)
    extract = _proportional_integral_extract
    clip = _proportional_integral_clip
    return Controller(init=init, apply=apply, extract=extract, clip=clip)


def _proportional_integral_apply(
    state: tuple[float, float],
    /,
    error_normalised,
    *,
    error_contraction_rate,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> tuple[float, float]:
    dt_proposed, error_norm_previously_accepted = state
    n1 = power_integral_unscaled / error_contraction_rate
    n2 = power_proportional_unscaled / error_contraction_rate

    a1 = (1.0 / error_normalised) ** n1
    a2 = (error_norm_previously_accepted / error_normalised) ** n2
    scale_factor_unclipped = safety * a1 * a2

    scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
    scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
    error_norm_previously_accepted = np.where(
        error_normalised <= 1.0, error_normalised, error_norm_previously_accepted
    )

    dt_proposed = scale_factor * dt_proposed
    return dt_proposed, error_norm_previously_accepted


def _proportional_integral_init(dt0, /):
    return dt0, 1.0


def _proportional_integral_clip(
    state: tuple[float, float], /, t, t1
) -> tuple[float, float]:
    dt_proposed, error_norm_previously_accepted = state
    dt = dt_proposed
    dt_clipped = np.minimum(dt, t1 - t)
    return dt_clipped, error_norm_previously_accepted


def _proportional_integral_extract(state: tuple[float, float], /):
    dt_proposed, _error_norm_previously_accepted = state
    return dt_proposed


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
    return np.minimum(dt, t1 - t)


def _no_clip(dt, /, *_args, **_kwargs):
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

    scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
    scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
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


tree_util.register_pytree_node(Controller, _flatten, _unflatten)
