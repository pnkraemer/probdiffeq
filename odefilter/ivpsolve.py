"""Solve initial value problems.

There are many ways to skin the cat, and there are even more ways
of stepping through time to solve an ODE.
Most libraries expose a single ``solve_ode()`` method.
``odefilters`` does not, there are differnet function calls for different
time-stepping modes and/or quantities of interest. We distinguish

### Adaptive time-stepping
* Terminal value simulation
* Checkpoint simulation
* Complete simulation (native python)
* Complete simulation (diffrax' bounded_while_loop)

### Constant time-stepping
* Fixed step-sizes
* Fixed evaluation grids

Why this distinction?

### Specialised solvers

If we know that only the terminal value is of interest,
we may build specialised ODE filters that compute this value,
and only this value, very efficiently.
In a more technical speak, we enter the world of Gaussian filters
and don't have to concern ourselves with smoothing mechanisms.
This avoids unnecessary matrix-matrix operations during the forward pass
and is generally a good idea if speed is required (which it always is).



### Specialised control flow

ODE solvers are iterative algorithms and heavily rely on control flow
such as while loops or scans.
But not all control flow is created equal.
Consider the most obvious example: while-loops
are not reverse-mode differentiable, and we need some tricks
to make it work (for the experts: we need to rely on continuous
sensitivity analysis to replace automatic differentiation).
But if we use a pre-determined grid, we can simply scan through time
and get all differentiation for free.
There are more of those subtleties, and it is simply easier to
split the simulations up into different functions.

### Code is easier to read
Most ``solve_ode()`` style codes are very involved and quite complex,
because they need to cover all eventualities of evaluation points (checkpoints),
events, constant or adaptive step-sizes, and so on, often with many
unused arguments depending on the function call.
In contrast, the simulation routines in ``odefilters`` are ~10 LoC and
all arguments are used (with some minor exceptions).




!!! warning "Initial value format"
    Functions in this module expect that the initial values are a tuple of arrays
    such that the vector field evaluates as
    ``vector_field(*initial_values, t, *parameters)``.
    This is different to most other ODE solver libraries, and done
    on purpose because higher-order ODEs are treated very similarly
    to first-order ODEs in this package.

"""


import jax
import jax.numpy as jnp


# todo: remove this and replace with jax.jit.
#  We need more transparency of what is static and what is not
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
        solver=solver,
    )
    return state.accepted


# todo: don't evaluate the ODE if the time-step has been clipped
def simulate_checkpoints(vector_field, initial_values, *, ts, solver, parameters=()):
    """Solve an IVP and return the solution at checkpoints."""
    _verify_not_scalar(initial_values=initial_values)

    # Include the parameters into the vector field.
    # This is done inside this function, because we don't want to
    # re-compile the whole solve if a parameter changes.
    def vf(*ys, t):
        return vector_field(*ys, t, *parameters)

    def advance_to_next_checkpoint(s, t_next):
        state_ = _advance_ivp_solution_adaptively(
            state0=s,
            t1=t_next,
            vector_field=vf,
            solver=solver,
        )
        return state_, state_.accepted

    state0 = solver.init_fn(vector_field=vf, initial_values=initial_values, t0=ts[0])
    _, solution = jax.lax.scan(
        f=advance_to_next_checkpoint,
        init=state0,
        xs=ts[1:],
        reverse=False,
    )
    return solution


# todo: allow scalar problems.
#  There is no clear mechanism for the internals if the IVP is
#  scalar. Therefore, we don't allow them for now.
def _verify_not_scalar(initial_values):
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

    init_val = solver.reset_fn(state=state0)
    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=init_val,
    )
