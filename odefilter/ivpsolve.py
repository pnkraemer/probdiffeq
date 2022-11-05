"""ODE solver routines."""


import functools
import warnings

import jax
import jax.numpy as jnp

from odefilter import odefiltersolve, taylor
from odefilter.strategies import smoothers

# The high-level checkpoint-style routines

# todo: call the argument "p" instead of "parameters"
#  to match the signature of the vector field?


@functools.partial(jax.jit, static_argnums=[0])
def simulate_terminal_values(
    vector_field, initial_values, t0, t1, solver, parameters=(), **options
):
    """Simulate the terminal values of an initial value problem.

    Thin wrapper around :func:`odefiltersolve.odefilter_terminal_values`.
    """
    _assert_not_scalar(initial_values)
    _assert_tuple(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=solver.strategy.extrapolation.num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )

    return odefiltersolve.odefilter_terminal_values(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
        **options,
    )


@functools.partial(jax.jit, static_argnums=[0])
def simulate_checkpoints(
    vector_field, initial_values, ts, solver, parameters=(), **options
):
    """Solve an IVP and return the solution at checkpoints.

    Thin wrapper around :func:`odefiltersolve.odefilter_checkpoints`.
    """
    _assert_not_scalar(initial_values)
    _assert_tuple(initial_values)
    if isinstance(solver.strategy, smoothers.Smoother):
        msg = (
            "A conventional smoother cannot be used."
            "Did you mean ``smoothers.FixedPointSmoother()``?"
        )
        warnings.warn(msg)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=solver.strategy.extrapolation.num_derivatives + 1 - len(initial_values),
        t=ts[0],
        parameters=parameters,
    )

    return odefiltersolve.odefilter_checkpoints(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        parameters=parameters,
        **options,
    )


# Full solver routines


def solve(vector_field, initial_values, t0, t1, solver, parameters=(), **options):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_not_scalar(initial_values)
    _assert_tuple(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=solver.strategy.extrapolation.num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )

    return odefiltersolve.odefilter(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
        **options,
    )


def solve_fixed_grid(
    vector_field, initial_values, ts, solver, parameters=(), **options
):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_not_scalar(initial_values)
    _assert_tuple(initial_values)

    taylor_coefficients = taylor.taylor_mode_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=solver.strategy.extrapolation.num_derivatives + 1 - len(initial_values),
        t=ts[0],
        parameters=parameters,
    )

    return odefiltersolve.odefilter_fixed_grid(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        parameters=parameters,
        **options,
    )


def _assert_not_scalar(x, /):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    is_not_scalar = jax.tree_util.tree_map(lambda s: jnp.ndim(s) > 0, x)
    assert jax.tree_util.tree_all(is_not_scalar)


def _assert_tuple(x, /):
    """Verify the initial conditions a tuple of arrays.

    todo: allow other containers.
    """
    assert isinstance(x, tuple)
