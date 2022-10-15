"""ODE filter strategies.

By construction (extrapolate-correct, not correct-extrapolate)
the solution intervals are right-including, i.e. defined
on the interval $(t_0, t_1]$.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.tree_util

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FilterOutput(Generic[T]):
    """Filtering solution."""

    t: float
    u: Any
    filtered: T
    diffusion_sqrtm: float

    def tree_flatten(self):
        children = self.t, self.u, self.filtered, self.diffusion_sqrtm
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BackwardModel(Generic[T]):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: T

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Posterior(Generic[T]):
    """Markov sequences as smoothing solutions."""

    t: float
    u: Any
    filtered: T
    diffusion_sqrtm: float
    backward_model: BackwardModel[T]

    def tree_flatten(self):
        children = (
            self.t,
            self.u,
            self.filtered,
            self.diffusion_sqrtm,
            self.backward_model,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFilter:
    """Filter implementation with dynamic calibration (time-varying diffusion)."""

    implementation: Any

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()

        sol = self.implementation.extract_sol(rv=corrected)
        filtered = FilterOutput(t=t0, u=sol, filtered=corrected, diffusion_sqrtm=1.0)
        return filtered, error_estimate

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, *, state, info_op, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(state.t + dt, m_ext)

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
            t=state.t + dt, u=sol, filtered=corrected, diffusion_sqrtm=diffusion_sqrtm
        )
        return filtered, error_estimate

    @staticmethod
    def extract_fn(*, state):  # noqa: D102
        return state

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102
        dt = t - s0.t
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
            t=t, u=sol, filtered=extrapolated, diffusion_sqrtm=s1.diffusion_sqrtm
        )
        return s1, target_p

    @staticmethod
    def reset_at_checkpoint_fn(*, solution, accepted, t1):  # noqa: D102

        sol = DynamicFilter._reset_t1(state=solution, t1=t1)
        acc = DynamicFilter._reset_t1(state=accepted, t1=t1)
        return acc, sol

    @staticmethod
    def _reset_t1(*, state, t1):
        return FilterOutput(
            t=t1,
            u=state.u,
            filtered=state.filtered,
            diffusion_sqrtm=state.diffusion_sqrtm,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _DynamicSmootherCommon:

    implementation: Any

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
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
            t=t0,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        # no jax.lax.cond here, because we condition on the _shape_ of the array
        # which is available at compilation time already.
        do_backward_pass = state.filtered.mean.ndim == 3
        if do_backward_pass:
            return self._smooth(state)

        return state

    def _smooth(self, state):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.filtered)
        marginals = self.implementation.marginalise_backwards(
            init=init, backward_model=state.backward_model
        )
        sol = self.implementation.extract_sol(rv=marginals)
        return Posterior(
            t=state.t,
            u=sol,
            filtered=marginals,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )

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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicSmoother(_DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, *, state, info_op, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(state.t + dt, m_ext)

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
        backward_model = BackwardModel(transition=backward_op, noise=backward_noise)

        # Final observation
        corrected = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = Posterior(
            t=state.t + dt,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102
        rv0, diffsqrtm = s0.filtered, s1.diffusion_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, diffusion_sqrtm=diffsqrtm, t=t, t0=s0.t
        )
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffsqrtm, t=s1.t, t0=t
        )

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        s0 = Posterior(
            t=t,
            u=sol,
            filtered=extrapolated0,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model0,
        )

        s1 = Posterior(
            t=s1.t,
            u=sol,
            filtered=s1.filtered,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model1,
        )
        return s1, s0

    @staticmethod
    def reset_at_checkpoint_fn(*, solution, accepted, t1):  # noqa: D102
        sol = DynamicSmoother._reset_t1(state=solution, t1=t1)
        acc = DynamicSmoother._reset_t1(state=accepted, t1=t1)
        return acc, sol

    @staticmethod
    def _reset_t1(*, state, t1):
        return Posterior(
            t=t1,  # new (better safe than sorry...)
            u=state.u,
            filtered=state.filtered,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFixedPointSmoother(_DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, *, state, info_op, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(state.t + dt, m_ext)

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
            # bw_state=state.backward_model,
            # bw_init=bw_increment,
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = Posterior(
            t=state.t + dt,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102
        rv0, diffsqrtm = s0.filtered, s1.diffusion_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, diffusion_sqrtm=diffsqrtm, t=t, t0=s0.t
        )
        extrapolated1, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffsqrtm, t=s1.t, t0=t
        )

        # The interpolated variable is the solution at the checkpoint,
        # and we need to update the backward models by condensing the
        # model from the previous checkpoint to t0 with the newly acquired
        # model from t0 to t. This will imply a backward model from the
        # previous checkpoint to the current checkpoint.
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model,
            bw_state=backward_model0
            # bw_init=backward_model0, bw_state=s0.backward_model
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        s0 = Posterior(
            t=t,
            u=sol,
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
            t=s1.t,
            u=sol,
            filtered=s1.filtered,
            diffusion_sqrtm=diffsqrtm,
            backward_model=bw1,
        )
        return s1, s0

    @jax.jit
    def reset_at_checkpoint_fn(self, *, solution, accepted, t1):  # noqa: D102
        acc = self._reset_accepted(state=accepted, t1=t1)
        sol = DynamicFixedPointSmoother._reset_at_t1(state=solution, t1=t1)
        return acc, sol

    def _reset_accepted(self, *, state, t1):
        bw_noise = self.implementation.init_backward_noise(
            rv_proto=state.backward_model.noise
        )
        bw_transition = self.implementation.init_backward_transition()
        bw_identity = BackwardModel(transition=bw_transition, noise=bw_noise)
        return Posterior(
            t=t1,
            u=state.u,
            filtered=state.filtered,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=bw_identity,
        )

    @staticmethod
    def _reset_at_t1(*, state, t1):
        return Posterior(
            t=t1,
            u=state.u,
            filtered=state.filtered,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )
