r"""Taylor-expand the solution of an initial value problem (IVP)."""


from probdiffeq.backend import functools, ode
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import discrete


def make_runge_kutta_starter(dt, *, atol=1e-12, rtol=1e-10):
    """Create an estimator that uses a Runge-Kutta starter."""
    # If the accuracy of the initialisation is bad, play around with dt.
    return functools.partial(_runge_kutta_starter, dt0=dt, atol=atol, rtol=rtol)


# atol and rtol must be static bc. of jax.odeint...
@functools.partial(
    functools.jit, static_argnums=[0], static_argnames=["num", "atol", "rtol"]
)
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
    ts = np.linspace(t, t + dt0 * (k - 1), num=k, endpoint=True)
    ys = ode.odeint_and_save_at(vf, initial_values, save_at=ts, atol=atol, rtol=rtol)

    # Initial condition
    estimator = discrete.fixedpointsmoother_precon()
    rv_t0 = impl.ssm_util.standard_normal(num + 1, 1.0)
    conditional_t0 = impl.ssm_util.identity_conditional(num + 1)
    init = (rv_t0, conditional_t0)

    # Discretised prior
    discretise = impl.ssm_util.ibm_transitions(num, 1.0)
    ibm_transitions = functools.vmap(discretise)(np.diff(ts))

    # Generate an observation-model for the QOI
    # (1e-7 observation noise for nuggets and for reusing existing code)
    model_fun = functools.vmap(
        impl.hidden_model.conditional_to_derivative, in_axes=(None, 0)
    )
    models = model_fun(0, 1e-7 * np.ones_like(ts))

    # Run the preconditioned fixedpoint smoother
    (corrected, conditional), _ = discrete.estimate_fwd(
        ys,
        init=init,
        prior_transitions=ibm_transitions,
        observation_model=models,
        estimator=estimator,
    )
    initial = impl.conditional.marginalise(corrected, conditional)
    return tuple(impl.stats.mean(initial))
