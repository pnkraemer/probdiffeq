"""ODE filter strategies.

By construction (extrapolate-correct, not correct-extrapolate)
the solution intervals are right-including, i.e. defined
on the interval $(t_0, t_1]$.
"""
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.tree_util

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


class FilteringDistribution(Generic[T], eqx.Module):

    filtered: T
    diffusion_sqrtm: float


class BackwardModel(Generic[T], eqx.Module):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: T


class SmoothingPosterior(Generic[T], eqx.Module):
    """Markov sequences as smoothing solutions."""

    filtered: T
    diffusion_sqrtm: float
    backward_model: BackwardModel[T]


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

        filtered = FilteringDistribution(filtered=corrected, diffusion_sqrtm=1.0)
        return filtered, error_estimate

    def step_fn(self, *, state, vector_field, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=state.filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
        )

        # Final observation
        corrected = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        u = self.implementation.extract_u(rv=corrected)

        filtered = FilteringDistribution(
            filtered=corrected, diffusion_sqrtm=diffusion_sqrtm
        )
        return filtered, error_estimate, u

    @staticmethod
    def reset_fn(*, state):  # noqa: D102
        return state

    @staticmethod
    def extract_fn(*, state):  # noqa: D102
        return state

    def interpolate_fn(self, *, s0, s1, t0, t1, t):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            s0.filtered.mean, p=p, p_inv=p_inv
        )
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=s0.filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=s1.diffusion_sqrtm,  # right-including intervals
        )
        solution = FilteringDistribution(
            filtered=extrapolated, diffusion_sqrtm=s1.diffusion_sqrtm
        )
        u = self.implementation.extract_u(rv=extrapolated)
        return solution, u


class DynamicSmoother(eqx.Module):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    implementation: Any
    information: Any

    def init_fn(self, *, taylor_coefficients):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = BackwardModel(
            transition=backward_transition, noise=backward_noise
        )

        solution = SmoothingPosterior(
            filtered=corrected,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    def reset_fn(self, *, state):
        """Change the backward model back to the identity.

        Initialises a new fixed-point smoother.
        """
        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(
            rv_proto=state.backward_model.noise
        )

        backward_model = BackwardModel(
            transition=backward_transition, noise=backward_noise
        )
        return SmoothingPosterior(
            filtered=state.filtered,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=backward_model,
        )

    def step_fn(self, *, state, vector_field, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.filtered.cov_sqrtm_lower,
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

        # Condense backward models
        bw_increment = BackwardModel(transition=backward_op, noise=backward_noise)
        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment, bw_init=state.backward_model
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        # Return solution
        smoothing_solution = SmoothingPosterior(
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )
        return smoothing_solution, error_estimate, u

    def extract_fn(self, *, state):  # noqa: D102
        # todo: this function checks in which mode it has been called,
        #  which is quite a dirty implementation.
        #  it also bastardises the filter/smoother naming.

        # If there is something to smooth, go ahead:
        if state.filtered.mean.ndim == 3:
            init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.filtered)
            return self.implementation.marginalise_backwards(
                init=init, backward_model=state.backward_model
            )

        # Otherwise, we are still in filtering mode and simply return
        # the input.
        return state

    def interpolate_fn(self, *, s0, s1, t):
        raise NotImplementedError(s0)
