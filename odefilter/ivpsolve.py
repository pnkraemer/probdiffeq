"""Solve initial value problems."""


import equinox as eqx
import jax
import jax.numpy as jnp


# todo: remove this and replace with jax.jit.
#  We need more transparency of what is static and what is not
@eqx.filter_jit
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

    !!! danger "Initial value format"
        This function expects that the initial values are a tuple of arrays
        such that the vector field evaluates as
        ``vector_field(*initial_values, t, *parameters)``.
        This is different to most other ODE solver libraries, and done
        on purpose because higher-order ODEs are treated very similarly
        to first-order ODEs in this package.

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
    _verify_not_scalar(initial_values=initial_values)

    # Include the parameters into the vector field.
    # This is done inside this function, because we don't want to
    # re-compile the whole solve if a parameter changes.
    def vf(*ys, t):
        return vector_field(*ys, t, *parameters)

    state0 = solver.init_fn(vector_field=vf, initial_values=initial_values, t0=t0)

    state = _advance_ivp_solution_adaptively(
        state0=state0,
        t1=t1,
        vector_field=vf,
        step_fn=solver.step_fn,
    )
    return state.accepted


# todo: don't evaluate the ODE if the time-step has been clipped
def solve_checkpoints(vector_field, initial_values, *, ts, solver, parameters=()):
    """Solve an IVP and return the solution at checkpoints."""
    _verify_not_scalar(initial_values=initial_values)

    # Include the parameters into the vector field.
    # This is done inside this function, because we don't want to
    # re-compile the whole solve if a parameter changes.
    def vf(*ys, t):
        return vector_field(*ys, t, *parameters)

    def solve_for_next_checkpoint(s, t_next):
        state_ = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vf,
            step_fn=solver.step_fn,
        )
        return state_, state_.accepted

    state0 = solver.init_fn(vector_field=vf, initial_values=initial_values, t0=ts[0])
    _, solution = jax.lax.scan(
        f=solve_for_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return solution

    # Return the initial value along with the rest, because
    # it contains components of the filtering solution
    return extract_qoi_fn(*state_terminal), state_qoi, extract_qoi_fn(*init_val)


# todo: allow scalar problems.
#  There is no clear mechanism for the internals if the IVP is
#  scalar. Therefore, we don't allow them for now.
def _verify_not_scalar(initial_values):
    initial_value_is_not_scalar = jax.tree_util.tree_map(
        lambda x: jnp.ndim(x) > 0, initial_values
    )
    assert jax.tree_util.tree_all(initial_value_is_not_scalar)


def _advance_ivp_solution_adaptively(*, vector_field, t1, state0, step_fn):
    """Advance an IVP solution from an initial state to a terminal state."""

    # todo:
    #  we need a call to solver.(re)init_fn to allow the smoothers to reset their backward models

    def cond_fun(s):
        return s.accepted.t < t1

    def body_fun(s):
        return step_fn(state=s, vector_field=vector_field, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=state0,
    )
