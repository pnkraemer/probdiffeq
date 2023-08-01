"""Routines for estimating solutions of initial value problems."""

import functools
import warnings

import jax
import jax.numpy as jnp

from probdiffeq import _adaptive, _collocate, _markov, solution, taylor
from probdiffeq.backend import tree_array_util


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
    **adaptive_solver_options,
):
    """Simulate the terminal values of an initial value problem."""
    _assert_tuple(initial_values)

    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, while_loop_fn=while_loop_fn_per_step, **adaptive_solver_options
    )

    num_derivatives = solver.strategy.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )
    solution_t0 = solver.solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    save_at = jnp.asarray([t1])
    posterior, output_scale, num_steps = _collocate.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        t=solution_t0.t,
        posterior=solution_t0.posterior,
        output_scale=solution_t0.output_scale,
        num_steps=solution_t0.num_steps,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        parameters=parameters,
        while_loop_fn=while_loop_fn_temporal,
        interpolate=(solver.interpolate_fun, solver.right_corner_fun),
    )
    # "squeeze"-type functionality (there is only a single state!)
    squeeze_fun = functools.partial(jnp.squeeze, axis=0)
    posterior = jax.tree_util.tree_map(squeeze_fun, posterior)
    output_scale = jax.tree_util.tree_map(squeeze_fun, output_scale)
    num_steps = jax.tree_util.tree_map(squeeze_fun, num_steps)

    if solver.requires_rescaling:
        if output_scale.ndim > 0:
            output_scale = output_scale[-1] * jnp.ones_like(output_scale)
        posterior = posterior.scale_covariance(output_scale)

    # I think the user expects marginals, so we compute them here
    if isinstance(posterior, _markov.MarkovSeqRev):
        marginals = posterior.init
    else:
        marginals = posterior
    u = marginals.extract_qoi_from_sample(marginals.mean)
    return solution.Solution(
        t=t1,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
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
    **adaptive_solver_options,
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

    if not solver.strategy.is_suitable_for_save_at:
        msg = "Strategy {solver.strategy} cannot be used in save_at mode. "
        warnings.warn(msg)

    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, while_loop_fn=while_loop_fn_per_step, **adaptive_solver_options
    )

    t0 = save_at[0]
    num_derivatives = solver.strategy.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )
    solution_t0 = solver.solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    posterior, output_scale, num_steps = _collocate.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        t=solution_t0.t,
        posterior=solution_t0.posterior,
        output_scale=solution_t0.output_scale,
        num_steps=solution_t0.num_steps,
        save_at=save_at[1:],
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        parameters=parameters,
        while_loop_fn=while_loop_fn_temporal,
        interpolate=(solver.interpolate_fun, solver.right_corner_fun),
    )

    if solver.requires_rescaling:
        if output_scale.ndim > 0:
            output_scale = output_scale[-1] * jnp.ones_like(output_scale)
        posterior = posterior.scale_covariance(output_scale)

    # I think the user expects marginals, so we compute them here
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=solution_t0.posterior)
    marginals, posterior = _tmp

    u = marginals.extract_qoi_from_sample(marginals.mean)
    return solution.Solution(
        t=save_at,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
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
    **adaptive_solver_options,
):
    """Solve an initial value problem with a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    _assert_tuple(initial_values)
    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, **adaptive_solver_options
    )
    # todo: call solver.num_derivatives()?
    num_derivatives = solver.strategy.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=t0,
        parameters=parameters,
    )
    solution_t0 = solver.solution_from_tcoeffs(
        taylor_coefficients, t=t0, output_scale=output_scale
    )

    if dt0 is None:
        f, u0s = vector_field, initial_values
        nugget = propose_dt0_nugget
        dt0 = propose_dt0(f, u0s, t0=t0, parameters=parameters, nugget=nugget)

    t, posterior, output_scale, num_steps = _collocate.solve_with_python_while_loop(
        jax.tree_util.Partial(vector_field),
        t=solution_t0.t,
        posterior=solution_t0.posterior,
        output_scale=solution_t0.output_scale,
        num_steps=solution_t0.num_steps,
        t1=t1,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        parameters=parameters,
        interpolate=(solver.interpolate_fun, solver.right_corner_fun),
    )
    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = jnp.concatenate((jnp.atleast_1d(t0), t))

    if solver.requires_rescaling:
        if output_scale.ndim > 0:
            output_scale = output_scale[-1] * jnp.ones_like(output_scale)
        posterior = posterior.scale_covariance(output_scale)

    # I think the user expects marginals, so we compute them here
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=solution_t0.posterior)
    marginals, posterior = _tmp

    u = marginals.extract_qoi_from_sample(marginals.mean)
    return solution.Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_fixed_grid(
    vector_field,
    initial_values,
    grid,
    solver,
    output_scale=1.0,
    parameters=(),
    taylor_fn=taylor.taylor_mode_fn,
):
    """Solve an initial value problem on a fixed, pre-determined grid."""
    _assert_tuple(initial_values)

    # Initialise the Taylor series
    num_derivatives = solver.strategy.extrapolation.num_derivatives
    taylor_coefficients = taylor_fn(
        vector_field=jax.tree_util.Partial(vector_field),
        initial_values=initial_values,
        num=num_derivatives + 1 - len(initial_values),
        t=grid[0],
        parameters=parameters,
    )
    solution_t0 = solver.solution_from_tcoeffs(
        taylor_coefficients, t=grid[0], output_scale=output_scale
    )

    # Compute the solution
    posterior, output_scale, num_steps = _collocate.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        posterior=solution_t0.posterior,
        output_scale=solution_t0.output_scale,
        num_steps=solution_t0.num_steps,
        grid=grid,
        solver=solver,
        parameters=parameters,
    )

    if solver.requires_rescaling:
        if output_scale.ndim > 0:
            output_scale = output_scale[-1] * jnp.ones_like(output_scale)
        posterior = posterior.scale_covariance(output_scale)

    # I think the user expects marginals, so we compute them here
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=solution_t0.posterior)
    marginals, posterior = _tmp

    u = marginals.extract_qoi_from_sample(marginals.mean)
    return solution.Solution(
        t=grid,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def _userfriendly_output(*, posterior, posterior_t0):
    if isinstance(posterior, _markov.MarkovSeqRev):
        marginals = posterior.marginalise_backwards()
        marginal_t1 = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # We need to include the initial filtering solution (as an init)
        # Otherwise, information is lost, and we cannot, e.g., interpolate correctly.
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = _markov.MarkovSeqRev(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior


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
