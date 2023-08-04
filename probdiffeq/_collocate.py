"""Sequential collocation.

Sequentially (and often, adaptively) constrain a random process to an ODE.
"""

import jax

from probdiffeq.backend import tree_array_util


def solve_and_save_at(
    vector_field,
    t,
    posterior,
    output_scale,
    num_steps,
    *,
    save_at,
    adaptive_solver,
    dt0,
    parameters,
    while_loop_fn,
    interpolate,
):
    interpolate_fun, no_interpolate_fun = interpolate

    def advance(acc_prev_ctrl, t_next):
        # Advance until accepted.t >= t_next.
        # Note: This could already be the case and we may not loop (just interpolate)
        accepted, previous, control = _advance_ivp_solution_adaptively(
            *acc_prev_ctrl,
            t1=t_next,
            vector_field=vector_field,
            adaptive_solver=adaptive_solver,
            parameters=parameters,
            while_loop_fn=while_loop_fn,
        )

        # Either interpolate (t > t_next) or "finalise" (t == t_next)
        accepted, solution, previous = jax.lax.cond(
            accepted.t > t_next,
            interpolate_fun,
            lambda _, *a: no_interpolate_fun(*a),
            t_next,
            previous,
            accepted,
        )

        # Extract the solution
        (_t, *sol_solver), _sol_ctrl = adaptive_solver.extract(solution, control)
        return (accepted, previous, control), sol_solver

    acc, ctrl = adaptive_solver.init(
        t=t,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
        dt0=dt0,
    )
    init = (acc, acc, ctrl)
    _, solution = jax.lax.scan(f=advance, init=init, xs=save_at, reverse=False)
    return solution


def _advance_ivp_solution_adaptively(
    acc0,
    prev0,
    ctrl0,
    *,
    vector_field,
    t1,
    adaptive_solver,
    parameters,
    while_loop_fn,
):
    """Advance an IVP solution to the next state."""

    def cond_fun(s):
        acc, _, ctrl = s
        return acc.t < t1

    def body_fun(s):
        s0, _, c0 = s
        s1, c1 = adaptive_solver.rejection_loop(
            s0, c0, vector_field=vector_field, t1=t1, parameters=parameters
        )
        return s1, s0, c1

    return while_loop_fn(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(acc0, prev0, ctrl0),
    )


def solve_with_python_while_loop(*args, **kwargs):
    generator = _solution_generator(*args, **kwargs)
    return tree_array_util.tree_stack(list(generator))


def _solution_generator(
    vector_field,
    t,
    posterior,
    output_scale,
    num_steps,
    *,
    dt0,
    t1,
    adaptive_solver,
    parameters,
    interpolate,
):
    """Generate a probabilistic IVP solution iteratively."""
    interpolate_fun, no_interpolate_fun = interpolate

    accepted, control = adaptive_solver.init(
        t, posterior, output_scale, num_steps, dt0=dt0
    )
    while accepted.t < t1:
        previous = accepted
        accepted, control = adaptive_solver.rejection_loop(
            accepted, control, vector_field=vector_field, t1=t1, parameters=parameters
        )

        if accepted.t < t1:
            sol_solver, _sol_control = adaptive_solver.extract(accepted, control)
            yield sol_solver

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if accepted.t > t1:
        accepted, solution, previous = interpolate_fun(s1=accepted, s0=previous, t=t1)
    else:
        assert accepted.t == t1
        accepted, solution, previous = no_interpolate_fun(previous, accepted)

    sol_solver, _sol_control = adaptive_solver.extract(solution, control)
    yield sol_solver


def solve_fixed_grid(
    vector_field, posterior, output_scale, num_steps, *, grid, solver, parameters
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
