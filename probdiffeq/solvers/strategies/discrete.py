"""Discrete filtering and smoothing."""

from typing import Any

import jax

from probdiffeq.backend import containers
from probdiffeq.impl import impl


# todo: fixedpointsmoother and kalmanfilter should be estimate()
#  with two different methods. This saves a lot of code.
def fixedpointsmoother_precon(data, /, init, conditional, observation_model):
    """Fixedpoint smoothing with a preconditioned prior."""
    # Incorporate final data point
    information_terminal = _select((data, observation_model), idx_or_slice=-1)
    init = _fixedpoint_precon_initialise(init, *information_terminal)

    # Scan over the remaining data points
    information = _select((data, observation_model), idx_or_slice=slice(0, -1, 1))
    return jax.lax.scan(
        f=_fixedpoint_precon_step,
        init=init,
        xs=(conditional, *information),
        reverse=True,
    )


class _FixedPointState(containers.NamedTuple):
    rv: Any
    conditional: Any


def _fixedpoint_precon_initialise(init, data, model):
    rv, cond = init
    _observed, conditional = impl.conditional.revert(rv, model)
    corrected = impl.conditional.apply(data, conditional)
    return _FixedPointState(corrected, cond)


def _fixedpoint_precon_step(state, cond_and_data_and_obs):
    conditional, data, observation = cond_and_data_and_obs
    rv, conditional = state

    # Extrapolate
    rv, conditional_new = impl.conditional.revert(rv, conditional)
    conditional = impl.conditional.merge(conditional, conditional_new)

    # Correct
    observed, reverse = impl.conditional.revert(rv, observation)
    corrected = impl.conditional.apply(data, reverse)

    # Scan-compatible output
    state = _FixedPointState(corrected, conditional)
    return state, state


def kalmanfilter_reverse(data, /, init, conditional, observation_model):
    """Reverse-time Kalman filter."""
    # Incorporate final data point
    information_terminal = _select((data, observation_model), idx_or_slice=-1)
    init = _kalman_reverse_initialise(init, *information_terminal)

    # Scan over the remaining data points
    information = _select((data, observation_model), idx_or_slice=slice(0, -1, 1))
    return jax.lax.scan(
        f=_kalman_reverse_step,
        init=init,
        xs=(conditional, *information),
        reverse=True,
    )


def _select(tree, idx_or_slice):
    return jax.tree_util.tree_map(lambda s: s[idx_or_slice, ...], tree)


class _KalmanFilterState(containers.NamedTuple):
    rv: Any
    num_data_points: int
    logpdf: float


def _kalman_reverse_initialise(rv, data, model):
    observed, conditional = impl.conditional.revert(rv, model)
    corrected = impl.conditional.apply(data, conditional)
    logpdf = impl.stats.logpdf(data, observed)
    return _KalmanFilterState(corrected, 1.0, logpdf)


def _kalman_reverse_step(state, cond_and_data_and_obs):
    conditional, data, observation = cond_and_data_and_obs
    rv, num_data, logpdf = state

    # Extrapolate-correct
    rv = impl.conditional.marginalise(rv, conditional)
    observed, reverse = impl.conditional.revert(rv, observation)
    corrected = impl.conditional.apply(data, reverse)

    # Update logpdf
    logpdf_new = impl.stats.logpdf(data, observed)
    logpdf_mean = impl.ssm_util.update_mean(logpdf, logpdf_new, num_data)
    state = _KalmanFilterState(corrected, num_data + 1.0, logpdf_mean)

    # Scan-compatible output
    return state, state
