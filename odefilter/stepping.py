"""Time-stepping."""
import abc
from typing import Any, Callable, Union

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.tree_util

from odefilter import sqrtm
from odefilter.prob import ibm, rv


def odefilter_non_adaptive(*, derivative_init_fn, num_derivatives, information_fn):
    """EK0 solver."""
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    return _ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=DynamicIsotropicEKF0(
            a=a, q_sqrtm_upper=q_sqrtm.T, information_fn=information_fn
        ),
    )


class _ODEFilter(eqx.Module):
    """ODE filter."""

    derivative_init_fn: Callable
    backend: Any

    class State(eqx.Module):
        """State."""

        # Mandatory
        t: float
        u: Any
        error_estimate: Any

        backend: Any

    def init_fn(self, *, ivp):
        """Initialise the IVP solver state."""
        f, u0, t0 = ivp.vector_field, ivp.initial_values, ivp.t0

        taylor_coefficients = self.derivative_init_fn(
            vector_field=lambda *x: f(*x, ivp.t0, *ivp.parameters),
            initial_values=(u0,),
            num=self.backend.num_derivatives,
        )

        backend_state = self.backend.init_fn(taylor_coefficients=taylor_coefficients)

        return self.State(
            t=t0,
            u=u0,
            error_estimate=jnp.nan,
            backend=backend_state,
        )

    def step_fn(self, *, state, vector_field, dt0):
        """Perform a step."""

        def vf(y):
            return vector_field(y, state.t)

        backend_state, error_estimate, u = self.backend.step_fn(
            state=state.backend, vector_field=vf, dt=dt0
        )
        return self.State(
            t=state.t + dt0, u=u, error_estimate=error_estimate, backend=backend_state
        )


class DynamicIsotropicEKF0(eqx.Module):
    """EK0 for terminal-value simulation with an isotropic covariance \
     structure and dynamic (time-varying) calibration."""

    a: Any
    q_sqrtm_upper: Any

    information_fn: Callable

    @property
    def q_sqrtm_lower(self):
        """Lower square root matrix."""
        return self.q_sqrtm_upper.T

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    class State(eqx.Module):
        """State."""

        rv_corrected: Any
        rv_extrapolated: Any

    def init_fn(self, *, taylor_coefficients):
        """Initialise."""
        m0_corrected = jnp.stack(taylor_coefficients)
        if m0_corrected.ndim == 1:
            m0_corrected = m0_corrected[:, None]

        c_sqrtm0_corrected = jnp.zeros(
            (self.num_derivatives + 1, self.num_derivatives + 1)
        )
        rv_corrected = rv.Normal(mean=m0_corrected, cov_sqrtm_upper=c_sqrtm0_corrected)

        m0_extrapolated = jnp.zeros_like(m0_corrected)
        c_sqrtm0_extrapolated = jnp.eye(*c_sqrtm0_corrected.shape)
        rv_extrapolated = rv.Normal(
            mean=m0_extrapolated, cov_sqrtm_upper=c_sqrtm0_extrapolated
        )
        return self.State(rv_extrapolated=rv_extrapolated, rv_corrected=rv_corrected)

    def step_fn(self, *, state, vector_field, dt):
        """Step."""
        # Turn this into a state = update_fn() thingy?
        # Do we want an init() method, too? (We probably need one.)
        # Is this its own class then? If so, what are the state and the params?
        x = self._evaluate_and_extrapolate_fn(
            dt=dt, vector_field=vector_field, state=state
        )
        (bias, linear_fn), error_estimate, rv_extrapolated = x

        # Final observation
        s_sqrtm = linear_fn(rv_extrapolated.cov_sqrtm_upper.T)  # shape (n,)
        s = jnp.dot(s_sqrtm, s_sqrtm)
        rv_observed = rv.Normal(mean=bias, cov_sqrtm_upper=jnp.sqrt(s))
        g = (rv_extrapolated.cov_sqrtm_upper.T @ s_sqrtm.T) / s  # shape (n,)

        # Final correction
        m_cor = rv_extrapolated.mean - g[:, None] * rv_observed.mean[None, :]
        c_sqrtm_cor = rv_extrapolated.cov_sqrtm_upper.T - g[:, None] * s_sqrtm[None, :]
        rv_corrected = rv.Normal(mean=m_cor, cov_sqrtm_upper=c_sqrtm_cor.T)

        state_new = self.State(
            rv_extrapolated=rv_extrapolated, rv_corrected=rv_corrected
        )
        return state_new, error_estimate, jnp.squeeze(rv_corrected.mean[0])

    def _evaluate_and_extrapolate_fn(self, *, dt, vector_field, state):
        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        # Extract previous correction
        (m0, c_sqrtm0) = state.rv_corrected.mean, state.rv_corrected.cov_sqrtm_upper

        # Extrapolate the mean and linearise the differential equation.
        m_extrapolated = p[:, None] * (self.a @ (p_inv[:, None] * m0))
        bias, linear_fn = self.information_fn(f=vector_field, x=m_extrapolated)

        # Observe the error-free state and calibrate some parameters
        s_sqrtm_lower = linear_fn(p_inv[:, None] * self.q_sqrtm_lower)
        s = jnp.dot(s_sqrtm_lower, s_sqrtm_lower)
        residual_white = bias / jnp.sqrt(s)
        diffusion_sqrtm = jnp.sqrt(
            jnp.dot(residual_white, residual_white) / residual_white.size
        )
        error_estimate = dt * diffusion_sqrtm * jnp.sqrt(s)

        # Full extrapolation
        c_sqrtm_extrapolated_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * c_sqrtm0)).T,
            R2=diffusion_sqrtm * self.q_sqrtm_lower,
        ).T
        c_sqrtm_extrapolated = p[:, None] * c_sqrtm_extrapolated_p

        rv_extrapolated = rv.Normal(
            mean=m_extrapolated, cov_sqrtm_upper=c_sqrtm_extrapolated
        )
        return (bias, linear_fn), error_estimate, rv_extrapolated
