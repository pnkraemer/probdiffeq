"""Inference via smoothing."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.strategies import _common


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicSmoother(_common.DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying output-scale)."""

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = _common.BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        posterior = _common.MarkovSequence(
            init=corrected, backward_model=backward_model
        )
        sol = self.implementation.extract_sol(rv=corrected)

        solution = _common.Solution(
            t=t0,
            t_previous=-jnp.inf,
            u=sol,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=1.0,
            num_data_points=1.0,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @jax.jit
    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.posterior.init.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)

        output_scale_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate = dt * output_scale_sqrtm * error_estimate

        extrapolated = self._complete_extrapolation(
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext=m_ext,
            m_ext_p=m_ext_p,
            p=p,
            p_inv=p_inv,
            posterior_previous=state.posterior,
        )

        # Final observation
        _, (corrected, _) = self._final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected.init)
        smoothing_solution = _common.Solution(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            posterior=corrected,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=state.num_data_points + 1,
        )

        return smoothing_solution, error_estimate

    def _final_correction(self, *, extrapolated, linear_fn, m_obs):
        a, (corrected, b) = self.implementation.final_correction(
            extrapolated=extrapolated.init, linear_fn=linear_fn, m_obs=m_obs
        )
        corrected_seq = _common.MarkovSequence(
            init=corrected, backward_model=extrapolated.backward_model
        )
        return a, (corrected_seq, b)

    def _complete_extrapolation(
        self, *, output_scale_sqrtm, m0_p, m_ext, m_ext_p, p, p_inv, posterior_previous
    ):
        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=posterior_previous.init.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = _common.BackwardModel(transition=bw_op, noise=bw_noise)
        return _common.MarkovSequence(init=extrapolated, backward_model=backward_model)

    def _case_right_corner(self, *, s0, s1, t):  # s1.t == t

        accepted = self._duplicate_with_unit_backward_model(state=s1, t=t)
        previous = _common.Solution(
            t=t,
            t_previous=s0.t,
            u=s1.u,
            posterior=s1.posterior,
            marginals=None,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        solution = previous

        return accepted, solution, previous

    def _case_interpolate(self, *, s0, s1, t):
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        rv0, diffsqrtm = s0.posterior.init, s1.output_scale_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, output_scale_sqrtm=diffsqrtm, t=t, t0=s0.t
        )
        posterior0 = _common.MarkovSequence(
            init=extrapolated0, backward_model=backward_model0
        )

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=diffsqrtm, t=s1.t, t0=t
        )
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = _common.Solution(
            t=t,
            t_previous=s0.t,
            u=sol,
            posterior=posterior0,
            marginals=None,
            output_scale_sqrtm=diffsqrtm,
            num_data_points=s1.num_data_points,
        )
        previous = solution

        accepted = _common.Solution(
            t=s1.t,
            t_previous=t,
            u=s1.u,
            posterior=posterior1,
            marginals=s1.marginals,
            output_scale_sqrtm=diffsqrtm,
            num_data_points=s1.num_data_points,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, state_previous, t, state):
        acc, _sol, _prev = self._case_interpolate(t=t, s1=state, s0=state_previous)
        sol_marginal = self.implementation.marginalise_model(
            init=acc.marginals,
            linop=acc.posterior.backward_model.transition,
            noise=acc.posterior.backward_model.noise,
        )
        u = self.implementation.extract_sol(rv=sol_marginal)
        return u, sol_marginal


def _nan_like(*args):
    return jax.tree_util.tree_map(_nan_like_array, *args)


def _nan_like_array(*args):
    return jnp.nan * jnp.ones_like(*args)
