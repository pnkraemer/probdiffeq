r"""Taylor-expand the solution of an initial value problem (IVP)."""

import functools

import jax
import jax.experimental.jet
import jax.experimental.ode
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import discrete

# todo: split into subpackage


def make_runge_kutta_starter(*, dt=1e-6, atol=1e-12, rtol=1e-10):
    """Create an estimator that uses a Runge-Kutta starter."""
    return functools.partial(_runge_kutta_starter, dt0=dt, atol=atol, rtol=rtol)


# atol and rtol must be static bc. of jax.odeint...
@functools.partial(jax.jit, static_argnums=[0], static_argnames=["num", "atol", "rtol"])
def _runge_kutta_starter(vector_field, initial_values, /, num: int, t, dt0, atol, rtol):
    # todo [inaccuracy]: the initial-value uncertainty is discarded
    # todo [feature]: allow implementations other than IsoIBM?
    # todo [feature]: higher-order ODEs

    # Assertions and early exits

    if len(initial_values) > 1:
        raise ValueError("Higher-order ODEs are not supported at the moment.")

    if num == 0:
        return initial_values

    if num == 1:
        return *initial_values, vector_field(*initial_values, t=t)

    # Generate data

    def func(y, t):
        return vector_field(y, t=t)

    # todo: allow flexible "solve" method?
    k = num + 1  # important: k > num
    ts = jnp.linspace(t, t + dt0 * (k - 1), num=k, endpoint=True)
    ys = jax.experimental.ode.odeint(func, initial_values[0], ts, atol=atol, rtol=rtol)

    # Discretise the prior
    conditional_t0 = impl.ssm_util.identity_conditional(num + 1)
    rv_t0 = impl.ssm_util.standard_normal(num + 1, 1.0)
    discretise = impl.ssm_util.ibm_transitions(num, 1.0)
    conditional, preconditioner = jax.vmap(discretise)(jnp.diff(ts))

    # Generate an observation-model for the QOI
    # (1e-7 observation noise for nuggets and for reusing existing code)
    model_fun = jax.vmap(impl.ssm_util.conditional_to_derivative, in_axes=(None, 0))
    models = model_fun(0, 1e-7 * jnp.ones_like(ts))

    # Run the preconditioned fixedpoint smoother
    (corrected, conditional), _ = discrete.fixedpointsmoother_precon(
        ys,
        init=(rv_t0, conditional_t0),
        conditional=(conditional, preconditioner),
        observation_model=models,
    )
    initial = impl.conditional.marginalise(corrected, conditional)
    return tuple(impl.random.mean(initial))
