"""Inference via filters."""
import abc

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.strategies import _common


@jax.tree_util.register_pytree_node_class
class FilterStrategy(_common.Strategy):
    def init_posterior(self, *, corrected):
        return corrected

    def case_right_corner(self, *, s0, s1, t):  # s1.t == t
        accepted = _common.Solution(
            t=t,
            t_previous=s0.t,  # todo: wrong, but no one cares
            u=s1.u,
            marginals=s1.marginals,
            posterior=s1.posterior,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        solution, previous = accepted, accepted

        return accepted, solution, previous

    def case_interpolate(self, *, s0, s1, t):
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.

        dt = t - s0.t
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, cache = self.extrapolate_mean(posterior=s0.posterior, p=p, p_inv=p_inv)
        extrapolated = self.complete_extrapolation(
            m_ext,
            cache,
            posterior_previous=s0.posterior,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=s1.output_scale_sqrtm,  # right-including intervals
        )
        sol = self.implementation.extract_sol(rv=extrapolated)
        target_p = _common.Solution(
            t=t,
            t_previous=t,
            u=sol,
            marginals=None,  # todo: what should happen here???
            posterior=extrapolated,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        return s1, target_p, target_p

    def offgrid_marginals(self, *, state_previous, t, state):
        _acc, sol, _prev = self.case_interpolate(t=t, s1=state, s0=state_previous)
        return sol.u, sol.posterior

    def marginals(self, *, posterior):
        return posterior

    def marginals_terminal_value(self, *, posterior):
        return posterior

    def scale_marginals(self, marginals, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=marginals, scale_sqrtm=output_scale_sqrtm
        )

    def scale_posterior(self, posterior, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=posterior, scale_sqrtm=output_scale_sqrtm
        )

    def extract_sol(self, x, /):
        return self.implementation.extract_sol(rv=x)

    def extrapolate_mean(self, *, posterior, p_inv, p):
        m_ext, *_ = self.implementation.extrapolate_mean(
            posterior.mean, p=p, p_inv=p_inv
        )
        return m_ext, ()

    def complete_extrapolation(
        self, m_ext, _cache, *, output_scale_sqrtm, posterior_previous, p, p_inv
    ):
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=posterior_previous.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
        )
        return extrapolated

    def final_correction(self, *, extrapolated, linear_fn, m_obs):
        return self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )


# Todo: In its current form, wouldn't this be a template for a NonDynamicSolver()?
#  All the "filter" information is hidden in _complete_extrapolation(), isn't it?
@jax.tree_util.register_pytree_node_class
class DynamicFilter(_common.DynamicSolver):
    """Filter implementation (time-constant output-scale)."""

    def __init__(self, *, implementation):
        strategy = FilterStrategy(implementation=implementation)
        super().__init__(strategy=strategy)

    def tree_flatten(self):
        children = (self.strategy.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)


@jax.tree_util.register_pytree_node_class
class Filter(_common.NonDynamicSolver):
    """Filter implementation (time-constant output-scale)."""

    def __init__(self, *, implementation):
        strategy = FilterStrategy(implementation=implementation)
        super().__init__(strategy=strategy)

    def tree_flatten(self):
        children = (self.strategy.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)
