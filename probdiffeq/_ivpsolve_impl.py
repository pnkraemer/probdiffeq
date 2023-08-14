"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""
import functools

import jax

from probdiffeq import _advance
from probdiffeq.backend import tree_array_util


def solve_and_save_at(
    vector_field,
    t,
    initial_condition,
    *,
    save_at,
    adaptive_solver,
    dt0,
    interpolate,
):
    interpolate_fun, right_corner_fun = interpolate

    advance_func = functools.partial(
        _advance.advance_and_interpolate,  # vs clip_and_advance?
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
        interpolate_fun=interpolate_fun,
        right_corner_fun=right_corner_fun,
    )

    acc, ctrl = adaptive_solver.init(t, initial_condition, dt0=dt0)
    init = (acc, acc, ctrl)
    _, solution = jax.lax.scan(f=advance_func, init=init, xs=save_at, reverse=False)
    return solution


def solve_and_save_every_step(*args, **kwargs):
    generator = _solution_generator(*args, **kwargs)
    return tree_array_util.tree_stack(list(generator))


def _solution_generator(
    vector_field,
    t,
    initial_condition,
    *,
    dt0,
    t1,
    adaptive_solver,
    interpolate,
):
    """Generate a probabilistic IVP solution iteratively."""
    interpolate_fun, right_corner_fun = interpolate
    accepted, control = adaptive_solver.init(t, initial_condition, dt0=dt0)

    while accepted.t < t1:
        previous = accepted
        accepted, control = adaptive_solver.rejection_loop(
            accepted, control, vector_field=vector_field, t1=t1
        )

        if accepted.t < t1:
            sol_solver, _sol_control = adaptive_solver.extract(accepted, control)
            yield sol_solver

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if accepted.t > t1:
        accepted, solution, previous = interpolate_fun(s1=accepted, s0=previous, t=t1)
    else:
        assert accepted.t == t1
        accepted, solution, previous = right_corner_fun(previous, accepted)

    sol_solver, _sol_control = adaptive_solver.extract(solution, control)
    yield sol_solver


def solve_fixed_grid(vector_field, initial_condition, *, grid, solver):
    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        s_new = solver.step(
            state=s,
            vector_field=vector_field,
            dt=dt,
        )
        return (s_new, t_new), s_new

    t0 = grid[0]
    state0 = solver.init(t0, initial_condition)
    _, result_state = jax.lax.scan(f=body_fn, init=(state0, t0), xs=grid[1:])

    _t, solution = solver.extract(result_state)
    return solution
