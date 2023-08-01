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

    def extract(self, posterior: _strategy.State, /):
        t = posterior.t
        ssv = self.correction.extract(posterior.ssv, posterior.corr)
        solution = self.extrapolation.extract(ssv, posterior.extra)
        return t, solution

    def case_right_corner(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:  # s1.t == t
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - s0.t

        ssv, extra = self.extrapolation.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.complete(ssv, extra, output_scale=output_scale)

        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        extrapolated = _strategy.State(t=t, ssv=ssv, extra=extra, corr=corr_like)
        return _interp.InterpRes(
            accepted=s1, solution=extrapolated, previous=extrapolated
        )

    def offgrid_marginals(
        self, *, t, marginals, posterior, posterior_previous, t0, t1, output_scale
    ):
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, posterior),
            s0=self.init(t0, posterior_previous),
            output_scale=output_scale,
        )
        t, posterior = self.extract(sol)
        u = posterior.extract_qoi_from_sample(posterior.mean)
        return u, posterior
