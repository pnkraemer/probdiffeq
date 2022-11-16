"""ODE solver routines."""


import warnings

import jax

from odefilter import _odefiltersolve, taylor
from odefilter.strategies import smoothers

# The high-level checkpoint-style routines

# todo: call the argument "p" instead of "parameters"
#  to match the signature of the vector field?


def simulate_terminal_values(
    vector_field,
    initial_values,
    t0,
    t1,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Simulate the terminal values of an initial value problem."""
    _assert_tuple(initial_values)

    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )

    return _odefiltersolve.simulate_terminal_values(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
        **options,
    )


def simulate_and_save_at(
    vector_field,
    initial_values,
    ts,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an IVP and return the solution at checkpoints."""
    _assert_tuple(initial_values)

    if isinstance(solver.strategy, smoothers.Smoother):
        msg = (
            "A conventional smoother cannot be used."
            "Did you mean ``smoothers.FixedPointSmoother()``?"
        )
        warnings.warn(msg)

    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=ts[0],
        parameters=parameters,
    )

    return _odefiltersolve.simulate_and_save_at(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        parameters=parameters,
        **options,
    )


# Full solver routines


def solve(
    vector_field,
    initial_values,
    t0,
    t1,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_tuple(initial_values)

    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )

    return _odefiltersolve.solve(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
        **options,
    )


def solve_fixed_grid(
    vector_field,
    initial_values,
    ts,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an initial value problem.

    !!! warning
        Uses native python control flow.
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_tuple(initial_values)

    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=ts[0],
        parameters=parameters,
    )

    return _odefiltersolve.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        ts=ts,
        solver=solver,
        parameters=parameters,
        **options,
    )


def _assert_tuple(x, /):
    """Verify the initial conditions a tuple of arrays.

    todo: allow other containers.
    """
    assert isinstance(x, tuple)
