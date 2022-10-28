"""Inference via filters."""
import jax
import jax.tree_util

from odefilter.strategies import _strategy


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy):
    """Filter."""

    def init_posterior(self, *, taylor_coefficients):
        return self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):  # s1.t == t
        return p1, p1, p1

    def case_interpolate(self, *, p0, rv1, t0, t, t1, scale_sqrtm):
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.

        dt = t - t0
        linearisation_pt, cache = self.begin_extrapolation(posterior=p0, dt=dt)
        extrapolated = self.complete_extrapolation(
            linearisation_pt,
            cache,
            posterior_previous=p0,
            output_scale_sqrtm=scale_sqrtm,
        )
        return rv1, extrapolated, extrapolated

    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            rv1=marginals,
            p0=posterior_previous,
            t0=t0,
            t1=t1,
            scale_sqrtm=scale_sqrtm,
        )
        u = self.extract_sol_terminal_value(posterior=sol)
        return u, sol

    def sample(self, key, *, posterior, shape):
        raise NotImplementedError

    def marginals(self, *, posterior):
        return posterior

    def marginals_terminal_value(self, *, posterior):
        return posterior

    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=marginals, scale_sqrtm=output_scale_sqrtm
        )

    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=posterior, scale_sqrtm=output_scale_sqrtm
        )

    def extract_sol_terminal_value(self, *, posterior):
        return self.implementation.extract_sol(rv=posterior)

    def extract_sol_from_marginals(self, *, marginals):
        return self.implementation.extract_sol(rv=marginals)

    def begin_extrapolation(self, *, posterior, dt):
        return self.implementation.begin_extrapolation(posterior.mean, dt=dt)

    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        return self.implementation.complete_extrapolation(
            linearisation_pt=linearisation_pt,
            cache=cache,
            l0=posterior_previous.cov_sqrtm_lower,
            output_scale_sqrtm=output_scale_sqrtm,
        )

    def complete_correction(self, *, info_op, extrapolated, cache_obs):
        return self.implementation.complete_correction(
            info_op=info_op,
            extrapolated=extrapolated,
            cache=cache_obs,
        )
