"""Inference via smoothing."""

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.strategies import _interface

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


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
    marginals: T
    marginals_filtered: T
    backward_model: BackwardModel[T]

    diffusion_sqrtm: float

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
            self.u,
            self.marginals,
            self.marginals_filtered,
            self.backward_model,
            self.diffusion_sqrtm,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (
            t,
            t_previous,
            u,
            marginals,
            marginals_filtered,
            backward_model,
            diffusion_sqrtm,
        ) = children
        return cls(
            t=t,
            t_previous=t_previous,
            u=u,
            marginals=marginals,
            marginals_filtered=marginals_filtered,
            backward_model=backward_model,
            diffusion_sqrtm=diffusion_sqrtm,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _DynamicSmootherCommon(_interface.Strategy):
    @abc.abstractmethod
    def _case_interpolate(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_right_corner(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, info_op, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def init_fn(self, taylor_coefficients, t0):
        raise NotImplementedError

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        # todo: are we looping correctly?
        #  what does the backward transition at time t say?
        #  How to get from t to the previous t, right?

        # no jax.lax.cond here, because we condition on the _shape_ of the array
        # which is available at compilation time already.
        do_backward_pass = state.marginals_filtered.mean.ndim == 3
        if do_backward_pass:
            return self._smooth(state)

        return Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals_filtered=state.marginals_filtered,
            marginals=state.marginals_filtered,  # we are at the terminal state only
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )
        return state

    def _smooth(self, state):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.marginals_filtered)
        marginals = self.implementation.marginalise_backwards(
            init=init, backward_model=state.backward_model
        )
        sol = self.implementation.extract_sol(rv=marginals)
        return Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=sol,
            marginals_filtered=state.marginals_filtered,
            marginals=marginals,
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
            t_previous=t,  # identity transition: this is what it does...
            u=s0.u,
            marginals_filtered=s0.marginals_filtered,
            marginals=s0.marginals,
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
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, *, state, info_op, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.marginals_filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(state.t + dt, m_ext)

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
            marginals_filtered=corrected,
            marginals=None,
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
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
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
        solution = Posterior(
            t=t,
            t_previous=s0.t,
            u=sol,
            marginals_filtered=extrapolated0,
            marginals=None,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model0,
        )
        previous = solution

        accepted = Posterior(
            t=s1.t,
            t_previous=t,
            u=sol,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            diffusion_sqrtm=diffsqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFixedPointSmoother(_DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

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
            t_previous=t0,
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, *, state, info_op, dt):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.marginals_filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(state.t + dt, m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate *= dt

        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.marginals_filtered.cov_sqrtm_lower,
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
            t_previous=state.t_previous,  # condensing the models...
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    def _case_right_corner(self, s0, s1, t):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?

        backward_model1 = s1.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=backward_model1
        )
        backward_model1 = BackwardModel(transition=g0, noise=noise0)
        solution = Posterior(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            backward_model=backward_model1,
            diffusion_sqrtm=s1.diffusion_sqrtm,
        )

        accepted = self._duplicate_with_unit_backward_model(solution, t)
        previous = accepted

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
            rv=s0.marginals_filtered, diffusion_sqrtm=diffusion_sqrtm, t=t, t0=s0.t
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=bw0
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = Posterior(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=sol,
            marginals_filtered=extrapolated0,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model0,
        )

        # new model! no condensing...
        previous = self._duplicate_with_unit_backward_model(solution, t)

        # From t to s1.t
        extra1, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffusion_sqrtm, t=s1.t, t0=t
        )
        accepted = Posterior(
            t=s1.t,
            t_previous=t,  # new model! No condensing...
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous
