"""Forward-only estimation: filtering."""
from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.backend import containers
from probdiffeq.impl import impl
from probdiffeq.strategies import _common, strategy


def filter(extrapolation_factory, corr, calib, /):
    """Create a filter strategy."""
    extrapolation = extrapolation_factory.filter()
    extrapolation_repr = extrapolation_factory.string_repr()
    strategy_impl = strategy.Strategy(
        extrapolation,
        corr,
        string_repr=f"<Filter with {extrapolation_repr}, {corr}>",
        is_suitable_for_save_at=True,
        # Right-corner: use default
        impl_right_corner="default",
        # Filtering behaviour for interpolation
        impl_interpolate=_filter_interpolate,
        impl_offgrid_marginals=_filter_offgrid_marginals,
    )
    return strategy_impl, calib


def _filter_offgrid_marginals(
    t,
    *,
    marginals,
    output_scale,
    posterior,
    posterior_previous,
    t0,
    t1,
    init,
    interpolate,
    extract,
):
    _acc, sol, _prev = interpolate(
        t=t,
        s1=init(t1, posterior),
        s0=init(t0, posterior_previous),
        output_scale=output_scale,
    )
    t, posterior = extract(sol)
    u = impl.random.qoi(posterior)
    return u, posterior


def _filter_interpolate(t, *, output_scale, s0, s1, extrapolation):
    # A filter interpolates by extrapolating from the previous time-point
    # to the in-between variable. That's it.
    dt = t - s0.t
    hidden, extra = extrapolation.begin(s0.hidden, s0.aux_extra, dt=dt)
    hidden, extra = extrapolation.complete(hidden, extra, output_scale=output_scale)
    corr = jax.tree_util.tree_map(jnp.empty_like, s0.aux_corr)
    extrapolated = _common.State(t=t, hidden=hidden, aux_extra=extra, aux_corr=corr)
    return _interp.InterpRes(accepted=s1, solution=extrapolated, previous=extrapolated)


def kalman_reverse(data, /, init, conditional, observation_model):
    """Reverse-time Kalman filter."""
    # Incorporate final data point
    information_terminal = _select((data, observation_model), idx_or_slice=-1)
    init = _kalman_reverse_initialise(init, *information_terminal)

    # Scan over the remaining data points
    information = _select((data, observation_model), idx_or_slice=slice(0, -1, 1))
    return jax.lax.scan(
        f=_kalman_reverse_step,
        init=init,
        xs=(conditional,) + information,
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
    logpdf = impl.random.logpdf(data, observed)
    return _KalmanFilterState(corrected, 1.0, logpdf)


def _kalman_reverse_step(state, cond_and_data_and_obs):
    conditional, data, observation = cond_and_data_and_obs
    rv, num_data, logpdf = state

    # Extrapolate-correct
    rv = impl.conditional.marginalise(rv, conditional)
    observed, reverse = impl.conditional.revert(rv, observation)
    corrected = impl.conditional.apply(data, reverse)

    # Update logpdf
    logpdf_new = impl.random.logpdf(data, observed)
    logpdf_mean = impl.ssm_util.update_mean(logpdf, logpdf_new, num_data)
    state = _KalmanFilterState(corrected, num_data + 1.0, logpdf_mean)

    # Scan-compatible output
    return state, state
