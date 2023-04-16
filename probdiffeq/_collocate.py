"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""

from probdiffeq import _control_flow


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
    return adaptive_solver.extract(solution)


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
    return adaptive_solver.extract_at_terminal_values(solution)


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
    # todo: if we init() and extract() here, the duplicate_with_backward_model()
    #  (and even the case_right_corner()) functionality can be removed for good.
    #  open Q: we should probably use extract_at_terminal_values, right?
    #  if so, how do we go about MLE-Solver scaling and
    #  smoother marginalisation? one final init() + extract()?

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
    _dt, sol = adaptive_solver.extract(forward_solution)
    return sol


def _solution_generator(vector_field, *, state, t1, adaptive_solver, parameters):
    """Generate a probabilistic IVP solution iteratively."""
    # todo: adaptive_solver.solution_time(s) < t1?
    while state.solution.t < t1:
        yield state
        state = adaptive_solver.step(
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
        s_new = solver.step(
            state=s,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
        )
        return (s_new, t_new), (s_new, t_new)

    _, (result, _) = _control_flow.scan_with_init(
        f=body_fn, init=(state, t0), xs=grid[1:]
    )
    sol = solver.extract(result)
    return sol
