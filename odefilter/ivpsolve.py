"""Solve initial value problems."""


from functools import partial

import jax
import jax.numpy as jnp

from odefilter import _control_flow, taylor

# The high-level checkpoint-style routines


@partial(jax.jit, static_argnums=[0])
def simulate_terminal_values(
    vector_field, /, initial_values, *, t0, t1, solver, parameters=()
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
    return odefilter_terminal_values(
        vector_field,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_terminal_values(
    vector_field, /, *, taylor_coefficients, t0, t1, solver, parameters=()
):
    """Simulate the terminal values of an ODE with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    @jax.jit
    def vf(t, *ys):
        return vector_field(t, *ys, *parameters)

    state0 = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vf,
        solver=solver,
    )
    return solver.extract_fn(state=solution)


@partial(jax.jit, static_argnums=[0])
def simulate_checkpoints(vector_field, /, initial_values, *, ts, solver, parameters=()):
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

    return odefilter_checkpoints(
        vector_field,
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        parameters=parameters,
    )


@partial(jax.jit, static_argnums=[0])
def odefilter_checkpoints(
    vector_field, /, *, taylor_coefficients, ts, solver, parameters=()
):
    """Simulate checkpoints of an ODE solution with an ODE filter."""
    _assert_not_scalar(taylor_coefficients)

    @jax.jit
    def vf(t, *ys):
        return vector_field(t, *ys, *parameters)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vf,
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


def solve(vector_field, /, initial_values, *, t0, t1, solver, parameters=()):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    solution_gen = solution_generator(
        vector_field,
        initial_values=initial_values,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
    )
    return _control_flow.tree_stack([sol for sol in solution_gen])


def solution_generator(
    vector_field, /, initial_values, *, t0, t1, solver, parameters=()
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

    return odefilter_generator(
        vector_field,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
    )


def odefilter_generator(
    vector_field, /, taylor_coefficients, *, t0, t1, solver, parameters=()
):
    """Generate an ODE filter solution iteratively."""
    _assert_not_scalar(taylor_coefficients)

    @jax.jit
    def vf(t, *ys):
        return vector_field(t, *ys, *parameters)

    state = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    while state.accepted.t < t1:
        yield solver.extract_fn(state=state)
        state = solver.step_fn(state=state, vector_field=vf, t1=t1)

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


def _advance_ivp_solution_adaptively(*, vector_field, t1, state0, solver):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.accepted.t < t1

    def body_fun(s):
        return solver.step_fn(state=s, vector_field=vector_field, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
