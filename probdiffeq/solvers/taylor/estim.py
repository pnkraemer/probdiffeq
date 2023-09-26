r"""Taylor-expand the solution of an initial value problem (IVP)."""

import functools

import jax
import jax.experimental.jet
import jax.experimental.ode
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import discrete


def make_runge_kutta_starter(dt, *, atol=1e-12, rtol=1e-10):
    """Create an estimator that uses a Runge-Kutta starter."""
    # If the accuracy of the initialisation is bad, play around with dt.
    return functools.partial(_runge_kutta_starter, dt0=dt, atol=atol, rtol=rtol)


# atol and rtol must be static bc. of jax.odeint...
@functools.partial(jax.jit, static_argnums=[0], static_argnames=["num", "atol", "rtol"])
def _runge_kutta_starter(vf, initial_values, /, num: int, t, dt0, atol, rtol):
    # TODO [inaccuracy]: the initial-value uncertainty is discarded
    # TODO [feature]: allow implementations other than IsoIBM?
    # TODO [feature]: higher-order ODEs

    # Assertions and early exits

    if len(initial_values) > 1:
        msg = "Higher-order ODEs are not supported at the moment."
        raise ValueError(msg)

    if num == 0:
        return initial_values

    if num == 1:
        return *initial_values, vf(*initial_values, t)

    # Generate data

    # TODO: allow flexible "solve" method?
    k = num + 1  # important: k > num
    ts = jnp.linspace(t, t + dt0 * (k - 1), num=k, endpoint=True)
    ys = jax.experimental.ode.odeint(vf, initial_values[0], ts, atol=atol, rtol=rtol)

    # Initial condition
    estimator = discrete.fixedpointsmoother_precon()
    rv_t0 = impl.ssm_util.standard_normal(num + 1, 1.0)
    conditional_t0 = impl.ssm_util.identity_conditional(num + 1)
    init = (rv_t0, conditional_t0)

    # Discretised prior
    discretise = impl.ssm_util.ibm_transitions(num, 1.0)
    ibm_transitions = jax.vmap(discretise)(jnp.diff(ts))

    # Generate an observation-model for the QOI
    # (1e-7 observation noise for nuggets and for reusing existing code)
    model_fun = jax.vmap(impl.hidden_model.conditional_to_derivative, in_axes=(None, 0))
    models = model_fun(0, 1e-7 * jnp.ones_like(ts))
    print(ys)

    # Run the preconditioned fixedpoint smoother
    (corrected, conditional), _ = discrete.estimate_fwd(
        ys,
        init=init,
        prior_transitions=ibm_transitions,
        observation_model=models,
        estimator=estimator,
    )
    initial = impl.conditional.marginalise(corrected, conditional)
    print(corrected)
    print(conditional)
    print(initial)
    return tuple(impl.stats.mean(initial))
