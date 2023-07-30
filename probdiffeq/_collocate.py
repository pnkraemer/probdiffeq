"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""

import jax

from probdiffeq.backend import tree_array_util


def solve_and_save_at(
    vector_field,
    *,
    t,
    posterior,
    output_scale,
    num_steps,
    save_at,
    adaptive_solver,
    dt0,
    parameters,
    while_loop_fn,
):
    def advance(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vector_field,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
            while_loop_fn=while_loop_fn,
        )
        return s_next, s_next

    state0 = adaptive_solver.init(t, posterior, output_scale, num_steps, dt0=dt0)

    _, sol = jax.lax.scan(f=advance, init=state0, xs=save_at[1:], reverse=False)
    (_t, posterior, output_scale, num_steps), _sol_ctrl = adaptive_solver.extract(sol)
    return posterior, output_scale, num_steps


def simulate_terminal_values(
    vector_field,
    *,
    t,
    posterior,
    output_scale,
    num_steps,
    t1,
    adaptive_solver,
    parameters,
    dt0,
    while_loop_fn,
):
    state0 = adaptive_solver.init(t, posterior, output_scale, num_steps, dt0=dt0)
    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
        while_loop_fn=while_loop_fn,
    )
    (sol_solver, _sol_control) = adaptive_solver.extract(solution)
    return sol_solver


def _advance_ivp_solution_adaptively(
    *,
    vector_field,
    t1,
    state0,
    adaptive_solver,
    parameters,
    while_loop_fn,
):
    """Advance an IVP solution to the next state."""

    def cond_fun(s):
        # todo: adaptive_solver.solution_time(s) < t1?
        return s.solution.t < t1

    def body_fun(s):
        state = adaptive_solver.step(
            state=s,
            vector_field=vector_field,
            t1=t1,
            parameters=parameters,
        )
        return state

    sol = while_loop_fn(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
    return sol


def solve_with_python_while_loop(
    vector_field,
    *,
    t,
    posterior,
    output_scale,
    num_steps,
    t1,
    adaptive_solver,
    dt0,
    parameters,
):
    state = adaptive_solver.init(t, posterior, output_scale, num_steps, dt0=dt0)
    generator = _solution_generator(
        vector_field,
        state=state,
        t1=t1,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    forward_solution = tree_array_util.tree_stack(list(generator))
    sol_solver, _sol_control = adaptive_solver.extract(forward_solution)
    return sol_solver


def _solution_generator(vector_field, *, state, t1, adaptive_solver, parameters):
    """Generate a probabilistic IVP solution iteratively."""
    # todo: adaptive_solver.solution_time(s) < t1?
    while state.solution.t < t1:
        state = adaptive_solver.step(
            state=state,
            vector_field=vector_field,
            t1=t1,
            parameters=parameters,
        )
        yield state


def solve_fixed_grid(
    vector_field, *, posterior, output_scale, num_steps, grid, solver, parameters
):
    t0 = grid[0]
    state0 = solver.init(t0, posterior, output_scale, num_steps)

    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        s_new = solver.step(
            state=s,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
        )
        return (s_new, t_new), s_new

    _, result_state = jax.lax.scan(f=body_fn, init=(state0, t0), xs=grid[1:])

    _t, posterior, output_scale, num_steps = solver.extract(result_state)
    return posterior, output_scale, num_steps
