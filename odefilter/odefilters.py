"""ODE filter routines."""
from functools import partial  # noqa: F401

import jax.lax
import jax.numpy as jnp
import jax.tree_util

from odefilter import _adaptive, _control_flow

# todo: warn that taylor_coefficients might have
#  to be recomputed when the parameters change.
#  it is generally safer to use the ivpsolve.py methods.


@jax.jit
def odefilter_terminal_values(
    info, taylor_coefficients, t0, t1, solver, parameters, **options
):
    """Simulate the terminal values of an ODE with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    adaptive_solver = _adaptive.AdaptiveODEFilter(solver=solver, **options)

    state0 = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        info_op=info,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    return adaptive_solver.extract_terminal_value_fn(state=solution)


@jax.jit
def odefilter_checkpoints(info, taylor_coefficients, ts, solver, parameters, **options):
    """Simulate checkpoints of an ODE solution with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    adaptive_solver = _adaptive.AdaptiveODEFilter(solver=solver, **options)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            info_op=info,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
        )
        return s_next, s_next

    state0 = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=ts[0])

    _, solution = _control_flow.scan_with_init(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return adaptive_solver.extract_fn(state=solution)


def _advance_ivp_solution_adaptively(info_op, t1, state0, adaptive_solver, parameters):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.solution.t < t1

    def body_fun(s):
        state = adaptive_solver.step_fn(
            state=s, info_op=info_op, t1=t1, parameters=parameters
        )
        return state

    sol = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
    return sol


def odefilter(info_op, taylor_coefficients, t0, t1, solver, parameters, **options):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_not_scalar(taylor_coefficients)
    adaptive_solver = _adaptive.AdaptiveODEFilter(solver=solver, **options)

    state = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)
    generator = _odefilter_generator(
        info_op,
        state=state,
        t1=t1,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    forward_solution = _control_flow.tree_stack([sol for sol in generator])
    return adaptive_solver.extract_fn(state=forward_solution)


def _odefilter_generator(info_op, *, state, t1, adaptive_solver, parameters):
    """Generate an ODE filter solution iteratively."""
    while state.solution.t < t1:
        yield state
        state = adaptive_solver.step_fn(
            state=state, info_op=info_op, t1=t1, parameters=parameters
        )

    yield state


def _assert_not_scalar(x, /):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    is_not_scalar = jax.tree_util.tree_map(lambda x: jnp.ndim(x) > 0, x)
    assert jax.tree_util.tree_all(is_not_scalar)
