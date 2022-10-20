"""Inference via smoothing."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.strategies import _smoother_common, markov


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicSmoother(_smoother_common.DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = markov.BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        sol = self.implementation.extract_sol(rv=corrected)

        solution = markov.Posterior(
            t=t0,
            t_previous=-jnp.inf,
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @jax.jit
    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.marginals_filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate *= dt

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.marginals_filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = markov.BackwardModel(transition=bw_op, noise=bw_noise)

        # Final observation
        _, (corrected, _) = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = markov.Posterior(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    def _case_right_corner(self, *, s0, s1, t):  # s1.t == t

        accepted = self._duplicate_with_unit_backward_model(state=s1, t=t)
        previous = markov.Posterior(
            t=t,
            t_previous=s0.t,
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            diffusion_sqrtm=s1.diffusion_sqrtm,
            backward_model=s1.backward_model,
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

        rv0, diffsqrtm = s0.marginals_filtered, s1.diffusion_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, diffusion_sqrtm=diffsqrtm, t=t, t0=s0.t
        )
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffsqrtm, t=s1.t, t0=t
        )

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = markov.Posterior(
            t=t,
            t_previous=s0.t,
            u=sol,
            marginals_filtered=extrapolated0,
            marginals=None,  # todo: fill this value here already?
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model0,
        )
        previous = solution

        accepted = markov.Posterior(
            t=s1.t,
            t_previous=t,
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=s1.marginals,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, state_previous, t, state):
        acc, sol, _prev = self._case_interpolate(t=t, s1=state, s0=state_previous)
        sol_marginal = self.implementation.marginalise_model(
            init=acc.marginals,
            linop=acc.backward_model.transition,
            noise=acc.backward_model.noise,
        )
        u = self.implementation.extract_sol(rv=sol_marginal)
        return markov.Posterior(
            t=t,
            t_previous=state_previous.t,
            marginals_filtered=sol.marginals_filtered,
            marginals=sol_marginal,
            diffusion_sqrtm=acc.diffusion_sqrtm,
            u=u,
            # the values would be meaningless:
            backward_model=_nan_like(sol.backward_model),
        )


def _nan_like(*args):
    return jax.tree_util.tree_map(_nan_like_array, *args)


def _nan_like_array(*args):
    return jnp.nan * jnp.ones_like(*args)
