import jax

from probdiffeq.backend import control_flow


def advance_and_interpolate(
    acc_prev_ctrl,
    t_next,
    *,
    vector_field,
    adaptive_solver,
    interpolate_fun,
    right_corner_fun,
):
    # Advance until accepted.t >= t_next.
    # Note: This could already be the case and we may not loop (just interpolate)
    accepted, previous, control = _advance_ivp_solution_adaptively(
        *acc_prev_ctrl,
        t1=t_next,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    accepted, solution, previous = jax.lax.cond(
        accepted.t > t_next,
        interpolate_fun,
        lambda _, *a: right_corner_fun(*a),
        t_next,
        previous,
        accepted,
    )

    # Extract the solution
    (_t, solution_solver), _sol_ctrl = adaptive_solver.extract(solution, control)
    return (accepted, previous, control), solution_solver


def _advance_ivp_solution_adaptively(
    acc0,
    prev0,
    ctrl0,
    *,
    vector_field,
    t1,
    adaptive_solver,
):
    """Advance an IVP solution to the next state."""

    def cond_fun(s):
        acc, _, ctrl = s
        return acc.t < t1

    def body_fun(s):
        s0, _, c0 = s
        s1, c1 = adaptive_solver.rejection_loop(
            s0, c0, vector_field=vector_field, t1=t1
        )
        return s1, s0, c1

    init = (acc0, prev0, ctrl0)
    return control_flow.while_loop(cond_fun, body_fun, init=init)
