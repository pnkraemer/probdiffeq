"""ODE solver routines."""


import functools

import jax
import jax.numpy as jnp

from odefilter import odefilters, taylor

# The high-level checkpoint-style routines

# todo: call the argument "p" instead of "parameters"
#  to match the signature of the vector field?


@functools.partial(jax.jit, static_argnums=[0, 5])
def simulate_terminal_values(
    vector_field, initial_values, t0, t1, solver, info_op, parameters=(), **options
):
    """Simulate the terminal values of an initial value problem.

    Thin wrapper around :func:`odefilter_terminal_values`.
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(*x, t=t0, p=parameters),
        initial_values=initial_values,
        num=solver.implementation.num_derivatives,
    )

    info_op_curried = info_op(vector_field)
    info_op_partial = functools.partial(info_op_curried, p=parameters)

    return odefilters.odefilter_terminal_values(
        info_op_partial,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        **options,
    )


@functools.partial(jax.jit, static_argnums=[0, 4])
def simulate_checkpoints(
    vector_field, initial_values, ts, solver, info_op, parameters=(), **options
):
    """Solve an IVP and return the solution at checkpoints.

    Thin wrapper around :func:`odefilter_checkpoints`.
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(*x, t=ts[0], p=parameters),
        initial_values=initial_values,
        num=solver.implementation.num_derivatives,
    )

    info_op_curried = info_op(vector_field)
    info_op_partial = functools.partial(info_op_curried, p=parameters)

    return odefilters.odefilter_checkpoints(
        info_op_partial,
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        **options,
    )


# Full solver routines


def solve(
    vector_field, initial_values, t0, t1, solver, info_op, parameters=(), **options
):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.

    !!! warning
        The parameters are essentially static. Why?
        Because we use ``lambda t, y: f(t, y, p)``-style implementations
        and pass this lambda function to lower-level implementations,
        which have static "fun" arguments. Since we cannot jit this function.
        the lower-level stuff must recompile... :(
    """
    _assert_not_scalar(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=lambda *x: vector_field(*x, t=t0, p=parameters),
        initial_values=initial_values,
        num=solver.implementation.num_derivatives,
    )

    # todo: because of this line, the function recompiles
    #  every single time it is called.
    #  This is because odefilter() marks the info_op as static, and because
    #  info_op() creates a new function every time it is called.
    #  Is it sufficient to make information operators cache output?
    info_op_curried = info_op(vector_field)

    # todo: this lambda function below is newly created at every
    #  call to solve() and therefore we recompile steps
    #  every single time. This is strange.
    info_op_partial = functools.partial(info_op_curried, p=parameters)
    return odefilters.odefilter(
        info_op_partial,
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        **options,
    )


def _assert_not_scalar(x, /):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    is_not_scalar = jax.tree_util.tree_map(lambda x: jnp.ndim(x) > 0, x)
    assert jax.tree_util.tree_all(is_not_scalar)
