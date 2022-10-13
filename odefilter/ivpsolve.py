"""Solve initial value problems."""


from functools import partial

import jax
import jax.numpy as jnp

from odefilter import _control_flow, taylor

# The high-level checkpoint-style routines


@partial(jax.jit, static_argnums=[0])
def simulate_terminal_values(
    vector_field, /, initial_values, *, t0, t1, solver, info_op, parameters=()
):
    """Simulate the terminal values of an initial value problem.

    Thin wrapper around :func:`odefilter_terminal_values`.
    """
    _assert_not_scalar(initial_values)

    @jax.jit
    def vf_auto_t0(*x):
        return vector_field(t0, *x, *parameters)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=vf_auto_t0,
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    def info_op_curried(t, *ys):
        def vf(*xs):
            return vector_field(t, *xs, *parameters)

        return info_op(vf, *ys)

    return odefilter_terminal_values(
        info_op_curried,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_terminal_values(info, /, *, taylor_coefficients, t0, t1, solver):
    """Simulate the terminal values of an ODE with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    state0 = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        info_op=info,
        solver=solver,
    )
    return solver.extract_fn(state=solution)


@partial(jax.jit, static_argnums=[0])
def simulate_checkpoints(
    vector_field, /, initial_values, *, ts, solver, info_op, parameters=()
):
    """Solve an IVP and return the solution at checkpoints.

    Thin wrapper around :func:`odefilter_checkpoints`.
    """
    _assert_not_scalar(initial_values)

    @jax.jit
    def vf_auto_t0(*x):
        return vector_field(ts[0], *x, *parameters)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=vf_auto_t0,
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    info_op_curried = _curry_info_op(
        *parameters, info_op=info_op, vector_field=vector_field
    )

    return odefilter_checkpoints(
        info_op_curried, taylor_coefficients=taylor_coefficients, ts=ts, solver=solver
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_checkpoints(info, /, *, taylor_coefficients, ts, solver):
    """Simulate checkpoints of an ODE solution with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            info_op=info,
            solver=solver,
        )
        return s_next, s_next

    state0 = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=ts[0])

    _, solution = _control_flow.scan_with_init(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return solver.extract_fn(state=solution)


# Full solver routines


def solve(*args, **kwargs):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    solution_gen = solution_generator(*args, **kwargs)
    return _control_flow.tree_stack([sol for sol in solution_gen])


def solution_generator(
    vector_field, /, initial_values, *, t0, t1, solver, info_op, parameters=()
):
    """Construct a generator of an IVP solution.

    Thin wrapper around :func:`odefilter_generator`.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_not_scalar(initial_values)

    @jax.jit
    def vf_auto_t0(*x):
        return vector_field(t0, *x, *parameters)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=vf_auto_t0,
        initial_values=initial_values,
        num=solver.strategy.implementation.num_derivatives,
    )

    def info_op_curried(t, *ys):
        def vf(*xs):
            return vector_field(t, *xs, *parameters)

        return info_op(vf, *ys)

    return odefilter_generator(
        info_op_curried,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
    )


def odefilter_generator(info_op, /, taylor_coefficients, *, t0, t1, solver):
    """Generate an ODE filter solution iteratively."""
    _assert_not_scalar(taylor_coefficients)

    state = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    while state.accepted.t < t1:
        yield solver.extract_fn(state=state)
        state = solver.step_fn(state=state, info_op=info_op, t1=t1)

    yield solver.extract_fn(state=state)


# Auxiliary routines


def _assert_not_scalar(x, /):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    is_not_scalar = jax.tree_util.tree_map(lambda x: jnp.ndim(x) > 0, x)
    assert jax.tree_util.tree_all(is_not_scalar)


def _advance_ivp_solution_adaptively(*, info_op, t1, state0, solver):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.solution.t < t1

    def body_fun(s):
        return solver.step_fn(state=s, info_op=info_op, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )


# todo: move this curry to information.py
# todo: include parameters here


def _curry_info_op(
    *parameters,
    info_op,
    vector_field,
):
    def info_op_curried(t, *ys):
        def vf(*xs):
            return vector_field(t, *xs, *parameters)

        return info_op(vf, *ys)

    return info_op_curried
