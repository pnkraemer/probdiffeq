"""Routines for estimating solutions of initial value problems."""


import warnings

import jax

from probdiffeq import _solution_routines_impl, taylor
from probdiffeq.strategies import smoothers

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

    return _solution_routines_impl.simulate_terminal_values(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        t1=t1,
        solver=solver,
        parameters=parameters,
        **options,
    )


def solve_and_save_at(
    vector_field,
    initial_values,
    save_at,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an initial value problem \
     and return the solution at a pre-determined grid.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
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
        t=save_at[0],
        parameters=parameters,
    )

    return _solution_routines_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        save_at=save_at,
        solver=solver,
        parameters=parameters,
        **options,
    )


# Full solver routines


def solve_with_python_while_loop(
    vector_field,
    initial_values,
    t0,
    t1,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an initial value problem with a native-Python while loop.

    !!! warning
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

    return _solution_routines_impl.solve_with_python_while_loop(
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
    grid,
    solver,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    **options,
):
    """Solve an initial value problem on a fixed, pre-determined grid."""
    _assert_tuple(initial_values)

    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=grid[0],
        parameters=parameters,
    )

    return _solution_routines_impl.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        taylor_coefficients=taylor_coefficients,
        grid=grid,
        solver=solver,
        parameters=parameters,
        **options,
    )


def _assert_tuple(x, /):
    """Verify that the initial conditions are a tuple of arrays.

    todo: allow other containers.
    """
    assert isinstance(x, tuple)
