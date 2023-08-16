import jax

from probdiffeq.backend import control_flow


def advance_and_interpolate(state, t_next, *, vector_field, adaptive_solver):
    # Advance until accepted.t >= t_next.
    # Note: This could already be the case and we may not loop (just interpolate)
    state = _advance_ivp_solution_adaptively(
        state,
        t1=t_next,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    state, solution = jax.lax.cond(
        state.t > t_next,
        adaptive_solver.interpolate_and_extract,
        lambda s, _t: adaptive_solver.right_corner_and_extract(s),
        state,
        t_next,
    )
    return state, solution
    # ((_t, solution), _ctrl, num) = solution
    # return state, solution
    #
    # # Extract the solution
    # (_t, solution_solver), _sol_ctrl = adaptive_solver.extract(solution, control)
    # return (accepted, previous, control), solution_solver
    #


def _advance_ivp_solution_adaptively(
    state0,
    *,
    vector_field,
    t1,
    adaptive_solver,
):
    """Advance an IVP solution to the next state."""

    def cond_fun(s):
        return s.t < t1

    def body_fun(s):
        return adaptive_solver.rejection_loop(s, vector_field=vector_field, t1=t1)

    return control_flow.while_loop(cond_fun, body_fun, init=state0)
