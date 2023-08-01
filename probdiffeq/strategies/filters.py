"""Forward-only estimation: filtering."""

import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.strategies import _strategy


def filter(extra, corr, calib, /):
    """Create a filter strategy."""
    factory, parameters = extra
    extrapolation = factory.filter(*parameters)
    strategy = _strategy.Strategy(
        extrapolation,
        corr,
        string_repr=f"<Filter with {factory.string_repr(*parameters)}, {corr}>",
        right_corner_fun=None,
        is_suitable_for_save_at=True,
        interpolate_fun=_filter_interpolate,
        offgrid_marginals_fun=_filter_offgrid_marginals,
    )
    return strategy, calib


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
    u = posterior.extract_qoi_from_sample(posterior.mean)
    return u, posterior


def _filter_interpolate(t, *, output_scale, s0, s1, extrapolation):
    # A filter interpolates by extrapolating from the previous time-point
    # to the in-between variable. That's it.
    dt = t - s0.t
    ssv, extra = extrapolation.begin(s0.ssv, s0.extra, dt=dt)
    ssv, extra = extrapolation.complete(ssv, extra, output_scale=output_scale)
    corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
    extrapolated = _strategy.State(t=t, ssv=ssv, extra=extra, corr=corr_like)
    return _interp.InterpRes(accepted=s1, solution=extrapolated, previous=extrapolated)
