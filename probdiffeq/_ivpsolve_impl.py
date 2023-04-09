"""Routines for estimating solutions of differential equations \
 constrained by Taylor-series initial information.

Essentially, these functions implement the IVP solution routines after
initialisation of the Taylor coefficients.
"""

from probdiffeq import _adaptive, _control_flow

# todo: rename to _collocate_seq.py ?
#  rationale: sequential collocation. Initial conditions are available.
#  We have an initial "posterior" (rename to "process"?) and a constraint,
#  and simulate the constrained posterior sequentially (and usually adaptively).


def simulate_terminal_values(
    vector_field,
    *,
    taylor_coefficients,
    t0,
    t1,
    solver,
    parameters,
    dt0,
    output_scale_sqrtm,
    while_loop_fn_temporal,
    while_loop_fn_per_step,
    **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, while_loop_fn=while_loop_fn_per_step, **options
    )

    state0 = adaptive_solver.init(
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        dt0=dt0,
        output_scale_sqrtm=output_scale_sqrtm,
    )

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
        while_loop_fn=while_loop_fn_temporal,
        output_scale_sqrtm=output_scale_sqrtm,
    )
    _dt, sol = adaptive_solver.extract_terminal_value_fn(solution)
    return sol


def solve_and_save_at(
    vector_field,
    taylor_coefficients,
    save_at,
    solver,
    dt0,
    parameters,
    while_loop_fn_temporal,
    while_loop_fn_per_step,
    **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, while_loop_fn=while_loop_fn_per_step, **options
    )

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vector_field,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
            while_loop_fn=while_loop_fn_temporal,
        )
        return s_next, s_next

    t0 = save_at[0]
    state0 = adaptive_solver.init(
        taylor_coefficients=taylor_coefficients, t0=t0, dt0=dt0
    )

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
    output_scale_sqrtm
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
            output_scale_sqrtm=output_scale_sqrtm,
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
    taylor_coefficients,
    t0,
    t1,
    solver,
    dt0,
    parameters,
    output_scale_sqrtm,
    **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver=solver, **options)

    state = adaptive_solver.init(
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        dt0=dt0,
        output_scale_sqrtm=output_scale_sqrtm,
    )
    generator = _solution_generator(
        vector_field,
        state=state,
        t1=t1,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
        output_scale_sqrtm=output_scale_sqrtm,
    )
    forward_solution = _control_flow.tree_stack(list(generator))
    _dt, sol = adaptive_solver.extract_fn(forward_solution)
    return sol


def _solution_generator(
    vector_field, *, state, t1, adaptive_solver, parameters, output_scale_sqrtm
):
    """Generate a probabilistic IVP solution iteratively."""
    while state.solution.t < t1:
        yield state
        state = adaptive_solver.step_fn(
            state=state,
            vector_field=vector_field,
            t1=t1,
            parameters=parameters,
            output_scale_sqrtm=output_scale_sqrtm,
        )

    yield state


def solve_fixed_grid(
    vector_field, taylor_coefficients, grid, solver, parameters, output_scale_sqrtm
):
    t0 = grid[0]
    state = solver.init_fn(
        taylor_coefficients=taylor_coefficients,
        t0=t0,
        output_scale_sqrtm=output_scale_sqrtm,
    )

    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        s_new = solver.step_fn(
            state=s,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
            output_scale_sqrtm=output_scale_sqrtm,
        )
        return (s_new, t_new), (s_new, t_new)

    _, (result, _) = _control_flow.scan_with_init(
        f=body_fn, init=(state, t0), xs=grid[1:]
    )
    sol = solver.extract_fn(result)
    return sol
