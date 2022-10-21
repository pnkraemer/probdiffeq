"""Inference via filters."""
import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.strategies import _interface

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FilteringSolution(Generic[T]):
    """Filtering solution."""

    t: float
    t_previous: float

    u: Any
    marginals: T

    diffusion_sqrtm: float
    num_data_points: int

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
            self.u,
            self.marginals,
            self.diffusion_sqrtm,
            self.num_data_points,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        t, t_previous, u, marginals, diffusion_sqrtm, num_data_points = children
        return cls(
            t=t,
            t_previous=t_previous,
            u=u,
            marginals=marginals,
            diffusion_sqrtm=diffusion_sqrtm,
            num_data_points=num_data_points,
        )

    def __len__(self):
        """Length of a solution object.

        Depends on the length of the underlying :attr:`t` attribute.
        """
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access the `i`-th sub-solution."""
        if jnp.ndim(self.t) < 1:
            raise ValueError(f"Solution object not batched :(, {jnp.ndim(self.t)}")
        if isinstance(item, tuple) and len(item) > jnp.ndim(self.t):
            # s[2, 3] forbidden
            raise ValueError(f"Inapplicable shape: {item, jnp.shape(self.t)}")
        return FilteringSolution(
            t=self.t[item],
            t_previous=self.t_previous[item],
            u=self.u[item],
            diffusion_sqrtm=self.diffusion_sqrtm[item],
            num_data_points=self.num_data_points[item],
            marginals=jax.tree_util.tree_map(lambda x: x[item], self.marginals),
        )

    def __iter__(self):
        """Iterate through the filtering solution."""
        for i in range(self.t.shape[0]):
            yield self[i]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _FilterCommon(_interface.Strategy):
    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()

        sol = self.implementation.extract_sol(rv=corrected)
        filtered = FilteringSolution(
            t=t0,
            t_previous=-jnp.inf,
            u=sol,
            marginals=corrected,
            diffusion_sqrtm=1.0,  # ?
            num_data_points=1.0,  # todo: make this an int
        )
        return filtered, error_estimate

    def _case_right_corner(self, *, s0, s1, t):  # s1.t == t
        accepted = FilteringSolution(
            t=t,
            t_previous=s0.t,  # todo: wrong, but no one cares
            u=s1.u,
            marginals=s1.marginals,
            diffusion_sqrtm=s1.diffusion_sqrtm,
            num_data_points=s1.num_data_points,
        )
        solution, previous = accepted, accepted

        return accepted, solution, previous

    def _case_interpolate(self, *, s0, s1, t):
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.

        dt = t - s0.t
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, *_ = self.implementation.extrapolate_mean(
            s0.marginals.mean, p=p, p_inv=p_inv
        )
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=s0.marginals.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=s1.diffusion_sqrtm,  # right-including intervals
        )
        sol = self.implementation.extract_sol(rv=extrapolated)
        target_p = FilteringSolution(
            t=t,
            t_previous=t,
            u=sol,
            marginals=extrapolated,
            diffusion_sqrtm=s1.diffusion_sqrtm,
            num_data_points=s1.num_data_points,
        )
        return s1, target_p, target_p

    def offgrid_marginals(self, *, state_previous, t, state):
        _acc, sol, _prev = self._case_interpolate(t=t, s1=state, s0=state_previous)
        return sol

    def _complete_extrapolation(self, *, diffusion_sqrtm, l0, m_ext, p, p_inv):
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=l0,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
        )
        return extrapolated

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, *, state):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFilter(_FilterCommon):
    """Filter implementation (time-constant diffusion)."""

    @jax.jit
    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, _, _ = self.implementation.extrapolate_mean(
            state.marginals.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate = error_estimate * dt * diffusion_sqrtm

        extrapolated = self._complete_extrapolation(
            diffusion_sqrtm=diffusion_sqrtm,
            l0=state.marginals.cov_sqrtm_lower,
            m_ext=m_ext,
            p=p,
            p_inv=p_inv,
        )

        # Final observation
        _, (corrected, _) = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        sol = self.implementation.extract_sol(rv=corrected)

        filtered = FilteringSolution(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            marginals=corrected,
            diffusion_sqrtm=diffusion_sqrtm,
            num_data_points=state.num_data_points + 1,
        )
        return filtered, error_estimate

    def extract_fn(self, *, state):  # noqa: D102
        return state

    def extract_terminal_value_fn(self, *, state):  # noqa: D102
        return state


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Filter(_FilterCommon):
    """Filter implementation with dynamic calibration (time-varying diffusion)."""

    @jax.jit
    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)
        # Extrapolate the mean
        m_ext, _, _ = self.implementation.extrapolate_mean(
            state.marginals.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate = error_estimate * dt * diffusion_sqrtm

        extrapolated = self._complete_extrapolation(
            diffusion_sqrtm=1.0,
            l0=state.marginals.cov_sqrtm_lower,
            m_ext=m_ext,
            p=p,
            p_inv=p_inv,
        )

        # Final observation
        observed, (corrected, _) = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Calibration of the global diffusion
        new_diffusion_sqrtm = self._update_diffusion_sqrtm(
            diffsqrtm=state.diffusion_sqrtm, n=state.num_data_points, obs=observed
        )

        # Extract and return solution
        sol = self.implementation.extract_sol(rv=corrected)
        filtered = FilteringSolution(
            t=state.t + dt,
            t_previous=state.t,
            u=sol,
            marginals=corrected,
            diffusion_sqrtm=new_diffusion_sqrtm,
            num_data_points=jnp.add(state.num_data_points, 1),
        )
        return filtered, error_estimate

    def _update_diffusion_sqrtm(self, *, diffsqrtm, n, obs):
        evidence_sqrtm = self.implementation.evidence_sqrtm(observed=obs)
        diffsqrtm_new = self.implementation.sum_sqrt_scalars(
            jnp.sqrt(n) * diffsqrtm, evidence_sqrtm
        )
        new_diffusion_sqrtm = jnp.reshape(diffsqrtm_new, ()) / jnp.sqrt(n + 1)
        return new_diffusion_sqrtm

    def extract_fn(self, *, state):  # noqa: D102
        diffusion_sqrtm = state.diffusion_sqrtm[-1] * jnp.ones_like(
            state.diffusion_sqrtm
        )
        marginals = self.implementation.scale_covariance(
            rv=state.marginals, scale_sqrtm=diffusion_sqrtm
        )
        return FilteringSolution(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals=marginals,
            diffusion_sqrtm=diffusion_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        diffusion_sqrtm = state.diffusion_sqrtm
        marginals = self.implementation.scale_covariance(
            rv=state.marginals, scale_sqrtm=diffusion_sqrtm
        )
        return FilteringSolution(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals=marginals,
            diffusion_sqrtm=diffusion_sqrtm,
            num_data_points=state.num_data_points,
        )
