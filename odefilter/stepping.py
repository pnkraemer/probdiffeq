"""Time-stepping."""
import abc
from typing import Any, Callable, Union

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.tree_util

from odefilter import backends, sqrtm
from odefilter.prob import ibm, rv


def ekf0_isotropic_dynamic(*, num_derivatives, information_fn):
    """EK0 solver."""
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    return backends.DynamicIsotropicEKF0(
        a=a, q_sqrtm_upper=q_sqrtm.T, information_fn=information_fn
    )


class ODEFilter(eqx.Module):
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
