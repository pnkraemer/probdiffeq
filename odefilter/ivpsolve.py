"""Solve initial value problems."""

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def simulate_terminal_values(ivp, /, *, solver):
    """Simulate the terminal values of an initial value problem."""
    state0 = solver.init_fn(ivp=ivp)

    state = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=ivp.t1,
        vector_field=lambda x, t: ivp.vector_field(x, t, *ivp.parameters),
        step_fn=solver.step_fn,
    )
    return state.accepted


def _advance_ivp_solution_adaptively(*, vector_field, t1, state0, step_fn):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.accepted.t < t1

    def body_fun(s):
        return step_fn(state=s, vector_field=vector_field, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
