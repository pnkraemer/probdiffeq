"""Forward-only estimation: filtering."""
import jax

from probdiffeq.strategies import _strategy


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy):
    def init_posterior(self, *, taylor_coefficients):
        return self.implementation.extrapolation.init_corrected(
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

    def extract_sol_terminal_value(self, *, posterior):
        return posterior.extract_qoi()

    def begin_extrapolation(self, *, posterior, dt):
        return self.implementation.extrapolation.begin_extrapolation(posterior, dt=dt)

    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        return self.implementation.extrapolation.complete_extrapolation(
            linearisation_pt=linearisation_pt,
            cache=cache,
            p0=posterior_previous,
            output_scale_sqrtm=output_scale_sqrtm,
        )

    def complete_correction(self, *, extrapolated, cache_obs):
        return self.implementation.correction.complete_correction(
            extrapolated=extrapolated, cache=cache_obs
        )
