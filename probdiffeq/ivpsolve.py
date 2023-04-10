"""Routines for estimating solutions of initial value problems."""


import warnings

import jax
import jax.numpy as jnp

from probdiffeq import _ivpsolve_impl, taylor
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
    output_scale=1.0,
    dt0=None,
    parameters=(),
    while_loop_fn_temporal=jax.lax.while_loop,
    while_loop_fn_per_step=jax.lax.while_loop,
    taylor_fn=taylor.taylor_mode_fn,
    propose_dt0_nugget=1e-5,
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
    sol = solver.empty_solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    # todo: should we already make the solver adaptive here?
    return _ivpsolve_impl.simulate_terminal_values(
        jax.tree_util.Partial(vector_field),
        solution=sol,
        t1=t1,
        solver=solver,
        parameters=parameters,
        dt0=dt0,
        while_loop_fn_temporal=while_loop_fn_temporal,
        while_loop_fn_per_step=while_loop_fn_per_step,
        **options,
    )


def solve_and_save_at(
    vector_field,
    initial_values,
    save_at,
    solver,
    output_scale=1.0,
    dt0=None,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    while_loop_fn_temporal=jax.lax.while_loop,
    while_loop_fn_per_step=jax.lax.while_loop,
    propose_dt0_nugget=1e-5,
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
        msg1 = "A conventional smoother cannot be used. "
        msg2 = "Did you mean ``smoothers.FixedPointSmoother()``?"
        warnings.warn(msg1 + msg2)

    t0 = save_at[0]
    num_derivatives = solver.strategy.implementation.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )
    sol = solver.empty_solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    return _ivpsolve_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        solution=sol,
        save_at=save_at,
        solver=solver,
        dt0=dt0,
        parameters=parameters,
        while_loop_fn_temporal=while_loop_fn_temporal,
        while_loop_fn_per_step=while_loop_fn_per_step,
        **options,
    )


# Full solver routines


def solve_with_python_while_loop(
    vector_field,
    initial_values,
    t0,
    t1,
    solver,
    output_scale=1.0,
    dt0=None,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
    propose_dt0_nugget=1e-5,
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
    sol = solver.empty_solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    return _ivpsolve_impl.solve_with_python_while_loop(
        jax.tree_util.Partial(vector_field),
        solution=sol,
        t1=t1,
        solver=solver,
        dt0=dt0,
        parameters=parameters,
        **options,
    )


def solve_fixed_grid(
    vector_field,
    initial_values,
    grid,
    solver,
    output_scale=1.0,
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
    sol = solver.empty_solution_from_tcoeffs(
        taylor_coefficients, t=grid[0], output_scale=output_scale
    )
    return _ivpsolve_impl.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        solution=sol,
        grid=grid,
        solver=solver,
        parameters=parameters,
        **options,
    )


def propose_dt0(
    vector_field, initial_values, /, t0, parameters, scale=0.01, nugget=1e-5
):
    """Propose an initial time-step."""
    u0, *_ = initial_values
    f0 = vector_field(*initial_values, t=t0, p=parameters)

    norm_y0 = jnp.linalg.norm(u0)
    norm_dy0 = jnp.linalg.norm(f0) + nugget

    return scale * norm_y0 / norm_dy0


def _assert_tuple(x, /):
    """Verify that the initial conditions are a tuple of arrays.

    todo: allow other containers.
    """
    assert isinstance(x, tuple)
