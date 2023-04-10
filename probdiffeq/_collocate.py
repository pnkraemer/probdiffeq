"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""

from probdiffeq import _control_flow


def simulate_terminal_values(
    vector_field,
    *,
    solution,
    t1,
    adaptive_solver,
    parameters,
    dt0,
    while_loop_fn,
):
    state0 = adaptive_solver.init(solution, dt0=dt0)
    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
        while_loop_fn=while_loop_fn,
    )
    _dt, sol = adaptive_solver.extract_terminal_values_fn(solution)
    return sol


def solve_and_save_at(
    vector_field,
    *,
    solution,
    save_at,
    adaptive_solver,
    dt0,
    parameters,
    while_loop_fn,
):
    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vector_field,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
            while_loop_fn=while_loop_fn,
        )
        return s_next, s_next

    state0 = adaptive_solver.init(solution, dt0=dt0)

    _, solution = _control_flow.scan_with_init(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=save_at[1:],
        reverse=False,
    )
    _dt, sol = adaptive_solver.extract_fn(solution)
    return sol


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
        return s.solution.t < t1

    def body_fun(s):
        state = adaptive_solver.step_fn(
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
    vector_field, *, solution, t1, adaptive_solver, dt0, parameters
):
    state = adaptive_solver.init(solution, dt0=dt0)
    generator = _solution_generator(
        vector_field,
        state=state,
        t1=t1,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    forward_solution = _control_flow.tree_stack(list(generator))
    _dt, sol = adaptive_solver.extract_fn(forward_solution)
    return sol


def _solution_generator(vector_field, *, state, t1, adaptive_solver, parameters):
    """Generate a probabilistic IVP solution iteratively."""
    while state.solution.t < t1:
        yield state
        state = adaptive_solver.step_fn(
            state=state,
            vector_field=vector_field,
            t1=t1,
            parameters=parameters,
        )

    yield state


def solve_fixed_grid(vector_field, *, solution, grid, solver, parameters):
    t0 = grid[0]
    state = solver.init(solution)

    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        s_new = solver.step_fn(
            state=s,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
        )
        return (s_new, t_new), (s_new, t_new)

    _, (result, _) = _control_flow.scan_with_init(
        f=body_fn, init=(state, t0), xs=grid[1:]
    )
    sol = solver.extract_fn(result)
    return sol
