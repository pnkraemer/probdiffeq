"""Solve initial value problems."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("ivp", "solver"))
def simulate_terminal_values(ivp, *, solver, solver_params):
    """Simulate the terminal values of an initial value problem."""
    state0 = solver.init_fn(ivp=ivp, params=solver_params)
    state = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=ivp.t1,
        ode_function=ivp.ode_function,
        step_fn=solver.step_fn,
        params=solver_params,
    )
    return state


def _advance_ivp_solution_adaptively(*, ode_function, t1, state0, step_fn, params):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.t < t1

    def body_fun(s):
        dt = jnp.minimum(t1 - s.t, s.dt_proposed)
        return step_fn(state=s, ode_function=ode_function, dt0=dt, params=params)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
