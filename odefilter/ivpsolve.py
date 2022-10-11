"""Solve initial value problems."""


import jax
import jax.numpy as jnp


def simulate_terminal_values(
    vector_field,
    initial_values,
    *,
    t0,
    t1,
    solver,
    parameters=(),
):
    """Simulate the terminal values of an initial value problem.

    Parameters
    ----------
    vector_field :
        ODE vector field. Signature ``vector_field(*initial_values, t, *parameters)``.
    initial_values :
        Initial values.
    t0 :
        Initial time.
    t1 :
        Terminal time.
    solver :
        ODE solver.
    parameters :
        ODE parameters.
    """
    _assert_not_scalar(initial_values=initial_values)

    def vf(*ys, t):
        return vector_field(*ys, t, *parameters)

    state0 = solver.init_fn(vector_field=vf, initial_values=initial_values, t0=t0)

    solution = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vf,
        solver=solver,
    )
    return solver.extract_fn(state=solution)


# todo: don't evaluate the ODE if the time-step has been clipped
def simulate_checkpoints(vector_field, initial_values, *, ts, solver, parameters=()):
    """Solve an IVP and return the solution at checkpoints."""
    _assert_not_scalar(initial_values=initial_values)

    def vf(*ys, t):
        return vector_field(*ys, t, *parameters)

    def advance_to_next_checkpoint(s, t_next):
        s_next = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vf,
            solver=solver,
        )
        return s_next, s_next

    state0 = solver.init_fn(vector_field=vf, initial_values=initial_values, t0=ts[0])
    _, solution = jax.lax.scan(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return solver.extract_fn(state=solution)


def _assert_not_scalar(initial_values):
    """Verify the initial conditions are not scalar.

    There is no clear mechanism for the internals if the IVP is
    scalar. Therefore, we don't allow them for now.

    todo: allow scalar problems.
    """
    initial_value_is_not_scalar = jax.tree_util.tree_map(
        lambda x: jnp.ndim(x) > 0, initial_values
    )
    assert jax.tree_util.tree_all(initial_value_is_not_scalar)


def _advance_ivp_solution_adaptively(*, vector_field, t1, state0, solver):
    """Advance an IVP solution from an initial state to a terminal state."""

    def cond_fun(s):
        return s.accepted.t < t1

    def body_fun(s):
        return solver.step_fn(state=s, vector_field=vector_field, t1=t1)

    # todo: this conflicts with the init_fn, doesnt it?
    #  There needs to be a smarter distinction.
    # state0 = solver.reset_fn(state=state0)
    state1 = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )

    def true_fn(s1):
        state = solver.interpolate_fn(state=s1, t=t1)
        return state

    def false_fn(s1):
        return s1

    pred = state1.accepted.t > t1
    return jax.lax.cond(pred, true_fn, false_fn, state1)
