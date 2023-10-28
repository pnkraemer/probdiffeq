"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""
import functools

import jax
import jax.numpy as jnp

from probdiffeq.backend import control_flow, tree_array_util


def solve_and_save_at(
    vector_field, t, initial_condition, *, save_at, adaptive_solver, dt0
):
    advance_func = functools.partial(
        _advance_and_interpolate,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    state = adaptive_solver.init(t, initial_condition, dt0=dt0, num_steps=0.0)
    _, solution = jax.lax.scan(f=advance_func, init=state, xs=save_at, reverse=False)
    return solution


def _advance_and_interpolate(state, t_next, *, vector_field, adaptive_solver):
    # Advance until accepted.t >= t_next.
    # Note: This could already be the case and we may not loop (just interpolate)
    def cond_fun(s):
        # Terminate the loop if
        # the difference from s.t to t_next is smaller than a constant factor
        # (which is a "small" multiple of the current machine precision)
        # or if s.t > t_next holds.
        return s.t + 10 * jnp.finfo(float).eps < t_next

    def body_fun(s):
        return adaptive_solver.rejection_loop(s, vector_field=vector_field, t1=t_next)

    state = control_flow.while_loop(cond_fun, body_fun, init=state)

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    state, solution = jax.lax.cond(
        state.t > t_next + 10 * jnp.finfo(float).eps,
        adaptive_solver.interpolate_and_extract,
        lambda s, _t: adaptive_solver.right_corner_and_extract(s),
        state,
        t_next,
    )
    return state, solution


def solve_and_save_every_step(*args, **kwargs):
    generator = _solution_generator(*args, **kwargs)
    return tree_array_util.tree_stack(list(generator))


def _solution_generator(
    vector_field, t, initial_condition, *, dt0, t1, adaptive_solver
):
    """Generate a probabilistic IVP solution iteratively."""
    state = adaptive_solver.init(t, initial_condition, dt0=dt0, num_steps=0)

    while state.t < t1:
        state = adaptive_solver.rejection_loop(state, vector_field=vector_field, t1=t1)

        if state.t < t1:
            solution = adaptive_solver.extract(state)
            yield solution

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if state.t > t1:
        _, solution = adaptive_solver.interpolate_and_extract(state, t=t1)
    else:
        _, solution = adaptive_solver.right_corner_and_extract(state)

    yield solution


def solve_fixed_grid(vector_field, initial_condition, *, grid, solver):
    def body_fn(s, dt):
        _error, s_new = solver.step(state=s, vector_field=vector_field, dt=dt)
        return s_new, s_new

    t0 = grid[0]
    state0 = solver.init(t0, initial_condition)
    _, result_state = jax.lax.scan(f=body_fn, init=state0, xs=jnp.diff(grid))
    return solver.extract(result_state)
