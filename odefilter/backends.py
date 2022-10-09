"""ODE filter backends."""
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import tree_util

from odefilter.prob import rv

NormalLike = TypeVar("RVLike", rv.Normal, rv.IsotropicNormal)
"""A type-variable to alias appropriate Normal-like random variables."""


class FilteringSolution(Generic[NormalLike], eqx.Module):
    """Filtering solution.

    By construction right-including, i.e. it defines the solution
    on the interval $(t_0, t_1]$.
    """

    corrected: NormalLike
    extrapolated: NormalLike


class BackwardModel(Generic[NormalLike], eqx.Module):
    """Backward model for posterior Gauss--Markov processes."""

    transition: Any
    noise: NormalLike


class SmoothingSolution(Generic[NormalLike], eqx.Module):
    """Filtering solution.

    By construction right-including, i.e. it defines the solution
    on the interval $(t_0, t_1]$.
    """

    filtering_solution: FilteringSolution[NormalLike]
    backward_model: BackwardModel[NormalLike]


class DynamicFilter(eqx.Module):
    """Filter implementation with dynamic calibration (time-varying diffusion)."""

    implementation: Any
    information: Any

    def init_fn(self, *, taylor_coefficients):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()

        extrapolated = tree_util.tree_map(jnp.empty_like, corrected)

        solution = FilteringSolution(extrapolated=extrapolated, corrected=corrected)
        return solution, error_estimate

    def step_fn(self, *, state, vector_field, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.corrected.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=state.corrected.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
        )

        # Final observation
        corrected, u = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        state_new = FilteringSolution(extrapolated=extrapolated, corrected=corrected)
        return state_new, error_estimate, u


class DynamicSmoother(eqx.Module):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    implementation: Any
    information: Any

    def init_fn(self, *, taylor_coefficients):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        empty = tree_util.tree_map(jnp.empty_like, corrected)

        filtering_solution = FilteringSolution(extrapolated=empty, corrected=corrected)

        backward_transition = self.implementation.init_backward_transition()
        backward_model = BackwardModel(transition=backward_transition, noise=empty)

        solution = SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    def step_fn(self, *, state, vector_field, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtering_solution.corrected.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.filtering_solution.corrected.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        extrapolated, (backward_noise, backward_op) = x

        # Final observation
        corrected, u = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        filtering_solution = FilteringSolution(
            extrapolated=extrapolated, corrected=corrected
        )

        # Condense backward models
        bw_increment = BackwardModel(transition=backward_op, noise=backward_noise)
        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment, bw_init=state.backward_model
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        smoothing_solution = SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_model=backward_model,
        )
        return smoothing_solution, error_estimate, u
