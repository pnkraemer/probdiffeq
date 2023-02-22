"""Routines for estimating solutions of differential equations \
 constrained by Taylor-series initial information.

Essentially, these functions implement the IVP solution routines after
initialisation of the Taylor coefficients.
"""

import jax

from probdiffeq import _adaptive, _control_flow


def simulate_terminal_values(
    vector_field, taylor_coefficients, t0, t1, solver, parameters, **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver=solver, **options)

    state0 = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    return adaptive_solver.extract_terminal_value_fn(state=solution)


def solve_and_save_at(
    vector_field, taylor_coefficients, save_at, solver, parameters, **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver=solver, **options)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vector_field,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
        )
        return s_next, s_next

    t0 = save_at[0]
    state0 = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    _, solution = _control_flow.scan_with_init(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=save_at[1:],
        reverse=False,
    )
    return adaptive_solver.extract_fn(state=solution)


def _advance_ivp_solution_adaptively(
    vector_field, t1, state0, adaptive_solver, parameters
):
    """Advance an IVP solution to the next state."""

    def cond_fun(s):
        return s.solution.t < t1

    def body_fun(s):
        state = adaptive_solver.step_fn(
            state=s, vector_field=vector_field, t1=t1, parameters=parameters
        )
        return state

    sol = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
    return sol


def solve_with_python_while_loop(
    vector_field, taylor_coefficients, t0, t1, solver, parameters, **options
):
    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver=solver, **options)

    state = adaptive_solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)
    generator = _solution_generator(
        vector_field,
        state=state,
        t1=t1,
        adaptive_solver=adaptive_solver,
        parameters=parameters,
    )
    forward_solution = _control_flow.tree_stack(list(generator))
    return adaptive_solver.extract_fn(state=forward_solution)


def _solution_generator(vector_field, *, state, t1, adaptive_solver, parameters):
    """Generate a probabilistic IVP solution iteratively."""
    while state.solution.t < t1:
        yield state
        state = adaptive_solver.step_fn(
            state=state, vector_field=vector_field, t1=t1, parameters=parameters
        )

    yield state


def solve_fixed_grid(vector_field, taylor_coefficients, grid, solver, parameters):
    # todo: annoying that the error estimate is not part of the state...
    t0 = grid[0]
    state, _ = solver.init_fn(taylor_coefficients=taylor_coefficients, t0=t0)

    def body_fn(carry, t_new):
        s, t_old = carry
        dt = t_new - t_old
        s_new, _ = solver.step_fn(
            state=s, vector_field=vector_field, dt=dt, parameters=parameters
        )
        return (s_new, t_new), (s_new, t_new)

    _, (result, _) = _control_flow.scan_with_init(
        f=body_fn, init=(state, t0), xs=grid[1:]
    )
    return solver.extract_fn(state=result)
