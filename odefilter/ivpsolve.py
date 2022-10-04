"""Solve initial value problems."""

import jax


def simulate_terminal_values(ivp, /, *, solver, solver_params):
    """Simulate the terminal values of an initial value problem."""
    state0 = solver.init_fn(ivp=ivp, params=solver_params)
    state = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=ivp.t1,
        ode_function=ivp.ode_function,
        perform_step_fn=solver.perform_step_fn,
        params=solver_params,
    )
    return state


def _advance_ivp_solution_adaptively(
    *, ode_function, t1, state0, perform_step_fn, params
):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.t < t1

    def body_fun(s):
        return perform_step_fn(s, ode_function=ode_function, t1=t1, params=params)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
