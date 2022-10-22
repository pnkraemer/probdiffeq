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


# @jax.tree_util.register_pytree_node_class
# class _FilterCommon(_common.Solver):
#
#     def _extrapolate_mean(self, *, posterior, p_inv, p):
#         m_ext, *_ = self.implementation.extrapolate_mean(
#             posterior.mean, p=p, p_inv=p_inv
#         )
#         return m_ext, ()
#
#     def _complete_extrapolation(
#         self, m_ext, _cache, *, output_scale_sqrtm, posterior_previous, p, p_inv
#     ):
#         extrapolated = self.implementation.complete_extrapolation(
#             m_ext=m_ext,
#             l0=posterior_previous.cov_sqrtm_lower,
#             p=p,
#             p_inv=p_inv,
#             output_scale_sqrtm=output_scale_sqrtm,
#         )
#         return extrapolated
#


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


# Todo: In its current form, wouldn't this be a template for a DynamicSolver()?
#  All the "filter" information is hidden in _complete_extrapolation(), isn't it?
# @jax.tree_util.register_pytree_node_class
class Filter:
    """Filter implementation with dynamic calibration (time-varying output-scale)."""

    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        # Pre-error-estimate steps
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)
        m_ext, cache = self._extrapolate_mean(
            posterior=state.posterior, p_inv=p_inv, p=p
        )

        # Linearise and estimate error
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)
        error_estimate, _ = self._estimate_error(linear_fn, m_obs, p)

        # Post-error-estimate steps
        extrapolated = self._complete_extrapolation(
            m_ext,
            cache,
            output_scale_sqrtm=1.0,
            posterior_previous=state.posterior,
            p=p,
            p_inv=p_inv,
        )  # This is the only filter/smoother consideration!

        # Complete step (incl. calibration!)
        output_scale_sqrtm, n = state.output_scale_sqrtm, state.num_data_points
        observed, (corrected, _) = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        new_output_scale_sqrtm = self._update_output_scale_sqrtm(
            diffsqrtm=output_scale_sqrtm, n=n, obs=observed
        )

        # Extract and return solution
        sol = self.implementation.extract_sol(rv=corrected)
        filtered = _common.Solution(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            marginals=None,
            posterior=corrected,
            output_scale_sqrtm=new_output_scale_sqrtm,
            num_data_points=n + 1,
        )
        return filtered, dt * error_estimate

    def _update_output_scale_sqrtm(self, *, diffsqrtm, n, obs):
        evidence_sqrtm = self.implementation.evidence_sqrtm(observed=obs)
        diffsqrtm_new = self.implementation.sum_sqrt_scalars(
            jnp.sqrt(n) * diffsqrtm, evidence_sqrtm
        )
        new_output_scale_sqrtm = jnp.reshape(diffsqrtm_new, ()) / jnp.sqrt(n + 1)
        return new_output_scale_sqrtm

    def extract_fn(self, *, state):  # noqa: D102
        output_scale_sqrtm = state.output_scale_sqrtm[-1] * jnp.ones_like(
            state.output_scale_sqrtm
        )

        # This would be different for different filters/smoothers, I suppose.
        # todo: this does not scale the marginal! Currently it is incorrect!
        marginals = self.implementation.scale_covariance(
            rv=state.posterior, scale_sqrtm=output_scale_sqrtm
        )
        return _common.Solution(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals=marginals,  # new!
            posterior=marginals,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        output_scale_sqrtm = state.output_scale_sqrtm
        marginals = self.implementation.scale_covariance(
            rv=state.posterior, scale_sqrtm=output_scale_sqrtm
        )
        return _common.Solution(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals=marginals,  # new!
            posterior=marginals,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )
