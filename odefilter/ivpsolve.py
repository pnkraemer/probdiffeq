"""Solve initial value problems."""


from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("f", "solver"))
def simulate_terminal_values(*, f, t0, t1, u0, solver, solver_params):

    state0 = solver.init_fn(f=f, t0=t0, u0=u0, params=solver_params)
    state = _solve_ivp_on_interval(
        f=f,
        t1=t1,
        state0=state0,
        perform_step_fn=solver.perform_step_fn,
        params=solver_params,
    )
    return state


def _solve_ivp_on_interval(*, f, t1, state0, perform_step_fn, params):
    """Solve an IVP adaptively on the interval (t0, t1).

    This function is used by the saveat() and the terminal_value() versions.
    """

    def cond_fun(s):
        return s.t < t1

    def body_fun(s):
        state = s
        return perform_step_fn(state, f=f, t1=t1, params=params)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
