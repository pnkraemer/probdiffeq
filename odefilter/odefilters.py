"""Time-stepping."""
from typing import Any, Callable

import equinox as eqx


class ODEFilterSolution(eqx.Module):
    """Solution object of an ODE filter.."""

    t: float
    u: Any
    error_estimate: Any

    posterior: Any


class ODEFilter(eqx.Module):
    """ODE filter."""

    taylor_series_init: Callable
    strategy: Any

    def init_fn(self, *, vector_field, initial_values, t0):
        """Initialise the IVP solver state."""

        def vf(*x):
            return vector_field(*x, t=t0)

        taylor_coefficients = self.taylor_series_init(
            vector_field=vf,
            initial_values=initial_values,
            num=self.strategy.implementation.num_derivatives,
        )

        posterior, error_estimate = self.strategy.init_fn(
            taylor_coefficients=taylor_coefficients
        )

        u0, *_ = initial_values
        return ODEFilterSolution(
            t=t0,
            u=u0,
            error_estimate=error_estimate,
            posterior=posterior,
        )

    def step_fn(self, *, state, vector_field, dt0):
        """Perform a step."""

        def vf(*y):
            return vector_field(*y, t=state.t)

        posterior, error_estimate, u = self.strategy.step_fn(
            state=state.posterior, vector_field=vf, dt=dt0
        )
        return ODEFilterSolution(
            t=state.t + dt0, u=u, error_estimate=error_estimate, posterior=posterior
        )

    def reset_fn(self, *, state):  # noqa: D102
        posterior_new = self.strategy.reset_fn(state=state.posterior)
        return ODEFilterSolution(
            t=state.t,
            u=state.u,
            error_estimate=state.error_estimate,
            posterior=posterior_new,
        )
