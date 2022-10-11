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


class FilterOutput(Generic[T], eqx.Module):
    """Filtering solution."""

    solution: T
    filtered: T
    diffusion_sqrtm: float


class BackwardModel(Generic[T], eqx.Module):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: T


class Posterior(Generic[T], eqx.Module):
    """Markov sequences as smoothing solutions."""

    solution: T
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

        sol = self.implementation.extract_sol(rv=corrected)
        filtered = FilterOutput(solution=sol, filtered=corrected, diffusion_sqrtm=1.0)
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
            linear_fn=linear_fn, m_obs=m_obs, p=p
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
        sol = self.implementation.extract_sol(rv=corrected)

        filtered = FilterOutput(
            solution=sol, filtered=corrected, diffusion_sqrtm=diffusion_sqrtm
        )
        return filtered, error_estimate

    @staticmethod
    def reset_fn(*, state):  # noqa: D102
        return state

    @staticmethod
    def extract_fn(*, state):  # noqa: D102
        return state.filtered, state.solution

    def interpolate_fn(self, *, s0, s1, t0, t1, t):  # noqa: D102
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, *_ = self.implementation.extrapolate_mean(
            s0.filtered.mean, p=p, p_inv=p_inv
        )
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=s0.filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=s1.diffusion_sqrtm,  # right-including intervals
        )
        sol = self.implementation.extract_sol(rv=extrapolated)
        target_p = FilterOutput(
            solution=sol, filtered=extrapolated, diffusion_sqrtm=s1.diffusion_sqrtm
        )
        return s1, target_p


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
            transition=backward_transition,
            noise=backward_noise,
        )
        sol = self.implementation.extract_sol(rv=corrected)

        solution = Posterior(
            solution=sol,
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
        raise RuntimeError
        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(
            rv_proto=state.backward_model.noise
        )
        backward_model = BackwardModel(
            transition=backward_transition, noise=backward_noise
        )
        return Posterior(
            solution=state.sol,
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
            linear_fn=linear_fn, m_obs=m_obs, p=p
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
        bw_increment = BackwardModel(transition=backward_op, noise=backward_noise)

        # Final observation
        corrected = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Condense backward models
        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=state.backward_model,
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = Posterior(
            solution=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        # todo: make those quantities property of the solution?
        #  that seems appropriate.
        return smoothing_solution, error_estimate

    def extract_fn(self, *, state):  # noqa: D102
        # todo: this function checks in which mode it has been called,
        #  which is quite a dirty implementation.
        #  it also bastardises the filter/smoother naming.

        # If there is something to smooth, go ahead:
        if state.filtered.mean.ndim == 3:
            init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.filtered)
            marginals = self.implementation.marginalise_backwards(
                init=init, backward_model=state.backward_model
            )
            sol = self.implementation.extract_sol(rv=marginals)
            return marginals, sol

        # Otherwise, we are still in filtering mode and simply return the input
        return state.filtered, state.solution

    def interpolate_fn(self, *, s0, s1, t0, t1, t):  # noqa: D102
        rv0, diffsqrtm = s0.filtered, s1.diffusion_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, diffusion_sqrtm=diffsqrtm, t=t, t0=t0
        )
        extrapolated1, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffsqrtm, t=t1, t0=t
        )

        # The interpolated variable is the solution at the checkpoint,
        # and we need to update the backward models by condensing the
        # model from the previous checkpoint to t0 with the newly acquired
        # model from t0 to t. This will imply a backward model from the
        # previous checkpoint to the current checkpoint.
        # noise0, g0 = self.implementation.condense_backward_models(
        #     bw_init=s0.backward_model, bw_state=backward_model0
        # )
        # backward_model0 = BackwardModel(transition=g0, noise=noise0)

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        s0 = Posterior(
            solution=sol,
            filtered=extrapolated0,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model0,
        )

        # We update the backward model from t_interp to t_accep
        # because the time-increment has changed.
        # In the next iteration, we iterate from t_accep to the next
        # checkpoint, and condense the backward models starting at the
        # backward model from t_accep, which must know how to get back
        # to the previous checkpoint.
        bw1 = backward_model1
        s1 = Posterior(
            solution=sol,
            filtered=s1.filtered,
            diffusion_sqrtm=diffsqrtm,
            backward_model=bw1,
        )
        return s1, s0

    def _interpolate_from_to_fn(self, rv, diffusion_sqrtm, t, t0):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            rv.mean, p=p, p_inv=p_inv
        )
        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=rv.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        extrapolated, (backward_noise, backward_op) = x

        backward_model = BackwardModel(transition=backward_op, noise=backward_noise)
        return extrapolated, backward_model
