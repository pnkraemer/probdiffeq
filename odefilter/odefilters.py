"""Time-stepping."""
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp


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

    def init_fn(self, *, vector_field, initial_values, t0):
        """Initialise the IVP solver state."""

        def vf(*x):
            return vector_field(*x, t=t0)

        taylor_coefficients = self.derivative_init_fn(
            vector_field=vf,
            initial_values=initial_values,
            num=self.backend.num_derivatives,
        )

        backend_state, error_estimate = self.backend.init_fn(
            taylor_coefficients=taylor_coefficients
        )

        u0, *_ = initial_values
        return self.State(
            t=t0,
            u=u0,
            error_estimate=error_estimate,
            backend=backend_state,
        )

    def step_fn(self, *, state, vector_field, dt0):
        """Perform a step."""

        def vf(*y):
            return vector_field(*y, t=state.t)

        backend_state, error_estimate, u = self.backend.step_fn(
            state=state.backend, vector_field=vf, dt=dt0
        )
        return self.State(
            t=state.t + dt0, u=u, error_estimate=error_estimate, backend=backend_state
        )
