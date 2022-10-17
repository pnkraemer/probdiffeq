"""Inference via filters."""
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
class DynamicFilter(_interface.Strategy):
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
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            filtered=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
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
            t=t,
            t_previous=t,
            u=sol,
            filtered=extrapolated,
            diffusion_sqrtm=s1.diffusion_sqrtm,
        )
        return s1, target_p, target_p
