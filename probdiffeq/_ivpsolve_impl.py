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
):
    advance_func = functools.partial(
        _advance.advance_and_interpolate,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    state = adaptive_solver.init(t, initial_condition, dt0=dt0, num_steps=0.0)
    _, solution = jax.lax.scan(f=advance_func, init=state, xs=save_at, reverse=False)
    (sol_solver), _sol_ctrl, num_steps = solution
    return sol_solver, num_steps


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
            sol_solver, _sol_control, nstep = adaptive_solver.extract(state)
            yield sol_solver, nstep

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if state.t > t1:
        _, (sol_solver, _, nstep) = adaptive_solver.interpolate_and_extract(state, t=t1)
    else:
        _, (sol_solver, _, nstep) = adaptive_solver.right_corner_and_extract(state)

    yield sol_solver, nstep


def solve_fixed_grid(vector_field, initial_condition, *, grid, solver):
    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        _, s_new = solver.step(
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
