"""Forward-only estimation: filtering."""

import jax
import jax.numpy as jnp

from probdiffeq import _interp
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
