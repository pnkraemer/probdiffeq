"""Solve initial value problems."""

from typing import Any, Callable, Iterable, NamedTuple, Optional, Union


class InitialValueProblem(NamedTuple):
    f: Callable
    """ODE vector field."""

    y0: Union[Any, Iterable[Any]]
    """Initial values."""

    p: Any
    """Parameters of the initial value problem."""

    t0: float
    """Initial time-point."""

    t1: float
    """Terminal time-point."""

    jac: Optional[Callable] = None
    """Jacobian of the vector field."""


from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("f", "solver"))
def simulate_terminal_values(*, f, t0, t1, u0, solver, **solver_kwargs):

    init_fn, perform_step_fn = solver
    state0 = init_fn(f=f, t0=t0, u0=u0, **solver_kwargs)
    perform_step_fn = partial(perform_step_fn, **solver_kwargs)
    state = _solve_ivp_on_interval(
        f=f,
        t1=t1,
        state0=state0,
        perform_step_fn=perform_step_fn,
    )
    return state


def _solve_ivp_on_interval(*, f, t1, state0, perform_step_fn):
    """Solve an IVP adaptively on the interval (t0, t1).

    This function is used by the saveat() and the terminal_value() versions.
    """

    @jax.jit
    def cond_fun(s):
        return s.t < t1

    @jax.jit
    def body_fun(s):
        state = s
        return perform_step_fn(state, f=f, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
