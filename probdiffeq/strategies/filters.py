"""Forward-only estimation: filtering."""
from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.strategies import _strategy


def filter(*impl):
    """Create a filter strategy."""
    extra, corr, calib = impl
    return _Filter(extra.filter, corr), calib


@jax.tree_util.register_pytree_node_class
class _Filter(_strategy.Strategy[_strategy.State, Any]):
    """Filter strategy."""

    def case_right_corner(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:  # s1.t == t
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        return _filter_interpolate(
            t, output_scale=output_scale, s0=s0, s1=s1, extrapolation=self.extrapolation
        )

    def offgrid_marginals(
        self, *, t, marginals, posterior, posterior_previous, t0, t1, output_scale
    ):
        return _filter_offgrid_marginals(
            t,
            output_scale=output_scale,
            posterior=posterior,
            posterior_previous=posterior_previous,
            t0=t0,
            t1=t1,
            init=self.init,
            interpolate=self.case_interpolate,
            extract=self.extract,
        )


def _filter_offgrid_marginals(
    t,
    *,
    output_scale,
    posterior,
    posterior_previous,
    t0,
    t1,
    init,
    interpolate,
    extract
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
