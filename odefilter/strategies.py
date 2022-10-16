"""ODE filter strategies.

By construction (extrapolate-correct, not correct-extrapolate)
the solution intervals are right-including, i.e. defined
on the interval $(t_0, t_1]$.
"""
import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FilterOutput(Generic[T]):
    """Filtering solution."""

    t: float
    t_previous: float
    u: Any
    filtered: T
    diffusion_sqrtm: float

    def tree_flatten(self):
        children = self.t, self.t_previous, self.u, self.filtered, self.diffusion_sqrtm
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
    t_previous: float
    u: Any
    filtered: T
    diffusion_sqrtm: float
    backward_model: BackwardModel[T]

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
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
class _StrategyCommon(abc.ABC):

    implementation: Any

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    @abc.abstractmethod
    def init_fn(self, *, taylor_coefficients, t0):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):
        raise NotImplementedError

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102

        # Cases to switch between
        branches = [
            self._case_right_corner,
            self._case_interpolate,
        ]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10
        is_in_between = jnp.logical_not(is_right_corner)

        index_as_array, *_ = jnp.where(
            jnp.asarray([is_right_corner, is_in_between]), size=1
        )
        index = jnp.reshape(index_as_array, ())
        return jax.lax.switch(index, branches, s0, s1, t)

    @abc.abstractmethod
    def _case_right_corner(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_interpolate(self, s0, s1, t):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFilter(_StrategyCommon):
    """Filter implementation with dynamic calibration (time-varying diffusion)."""

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()

        sol = self.implementation.extract_sol(rv=corrected)
        filtered = FilterOutput(
            t=t0, t_previous=-jnp.inf, u=sol, filtered=corrected, diffusion_sqrtm=1.0
        )
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

    def _case_right_corner(self, s0, s1, t):  # s1.t == t

        accepted = FilterOutput(
            t=t,
            t_previous=s0.t,  # todo: wrong, but no one cares
            u=s1.u,
            filtered=s1.filtered,
            diffusion_sqrtm=s1.diffusion_sqrtm,
        )
        solution, previous = accepted, accepted

        return accepted, solution, previous

    def _case_interpolate(self, s0, s1, t):

        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.

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
        return s1, target_p, target_p


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _DynamicSmootherCommon(_StrategyCommon):
    @abc.abstractmethod
    def _case_interpolate(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_right_corner(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, info_op, dt):
        raise NotImplementedError

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
            t_previous=-jnp.inf,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        # return state
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
            t_previous=state.t_previous,
            u=sol,
            filtered=marginals,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )

    def _duplicate_with_unit_backward_model(self, s0, t):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=s0.backward_model.noise
        )
        bw_model = BackwardModel(transition=bw_transition0, noise=bw_noise0)
        state1 = Posterior(
            t=t,
            t_previous=t,
            u=s0.u,
            filtered=s0.filtered,
            diffusion_sqrtm=s0.diffusion_sqrtm,
            backward_model=bw_model,
        )
        return state1

    def _interpolate_from_to_fn(self, rv, diffusion_sqrtm, t, t0):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            rv.mean, p=p, p_inv=p_inv
        )

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=rv.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
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

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)

        # Final observation
        corrected = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = Posterior(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    def _case_right_corner(self, s0, s1, t):  # s1.t == t

        accepted = self._duplicate_with_unit_backward_model(s1, t)
        previous = Posterior(
            t=t,
            t_previous=s0.t,
            u=s1.u,
            filtered=s1.filtered,
            diffusion_sqrtm=s1.diffusion_sqrtm,
            backward_model=s1.backward_model,
        )
        solution = previous

        return accepted, solution, previous

    def _case_interpolate(self, s0, s1, t):

        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.filtered,
        # but the backward transition is kept.

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
        solution = Posterior(
            t=t,
            t_previous=s0.t,
            u=sol,
            filtered=extrapolated0,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model0,
        )
        previous = solution

        accepted = Posterior(
            t=s1.t,
            t_previous=t,
            u=sol,
            filtered=s1.filtered,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous


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
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        # Return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = Posterior(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    def _case_right_corner(self, s0, s1, t):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=s0.filtered, diffusion_sqrtm=s1.diffusion_sqrtm, t=s1.t, t0=s0.t
        )

        # backward_model1 = s1.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=backward_model1
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)
        solution = Posterior(
            t=t,
            t_previous=s0.t,
            u=s1.u,
            filtered=s1.filtered,
            backward_model=backward_model0,
            diffusion_sqrtm=s1.diffusion_sqrtm,
        )

        accepted = self._duplicate_with_unit_backward_model(solution, t)
        previous = accepted

        print("Case: right corner", t)

        return accepted, solution, previous

    def _case_interpolate(self, s0, s1, t):  # noqa: D102
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # Use the s1.diffusion as a diffusion over the interval.
        # Filtering/smoothing solutions are right-including intervals.
        diffusion_sqrtm = s1.diffusion_sqrtm

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.filtered, diffusion_sqrtm=diffusion_sqrtm, t=t, t0=s0.t
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=bw0
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = Posterior(
            t=t,
            t_previous=s0.t,
            u=sol,
            filtered=extrapolated0,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model0,
        )

        #
        previous = self._duplicate_with_unit_backward_model(solution, t)

        # From t to s1.t
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffusion_sqrtm, t=s1.t, t0=t
        )
        accepted = Posterior(
            t=s1.t,
            t_previous=t,
            u=s1.u,
            filtered=s1.filtered,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous
