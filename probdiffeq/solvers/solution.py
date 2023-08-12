"""Interact with estimated solutions (on dense grids).

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

import jax
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers import markov
from probdiffeq.solvers.strategies import discrete

# todo: the functions in here should only depend on posteriors / strategies!


def offgrid_marginals_searchsorted(*, ts, solution, solver):
    """Compute off-grid marginals on a dense grid via jax.numpy.searchsorted.

    !!! warning
        The elements in ts and the elements in the solution grid must be disjoint.
        Otherwise, anything can happen and the solution will be incorrect.
        At the moment, we do not check this.

    !!! warning
        The elements in ts must be strictly in (t0, t1).
        They must not lie outside the interval, and they must not coincide
        with the interval boundaries.
        At the moment, we do not check this.
    """
    offgrid_marginals_vmap = jax.vmap(_offgrid_marginals, in_axes=(0, None, None))
    return offgrid_marginals_vmap(ts, solution, solver)


def _offgrid_marginals(t, solution, solver):
    # side="left" and side="right" are equivalent
    # because we _assume_ that the point sets are disjoint.
    index = jnp.searchsorted(solution.t, t)

    def _extract_previous(tree):
        return jax.tree_util.tree_map(lambda s: s[index - 1, ...], tree)

    def _extract(tree):
        return jax.tree_util.tree_map(lambda s: s[index, ...], tree)

    marginals = _extract(solution.marginals)

    # In the smoothing context:
    # Extract the correct posterior.init (aka the filtering solutions)
    # The conditionals are incorrect, but we don't really care about this.
    posterior = _extract(solution.posterior)
    posterior_previous = _extract_previous(solution.posterior)

    t0 = _extract_previous(solution.t)
    t1 = _extract(solution.t)
    output_scale = _extract(solution.output_scale)

    return solver.strategy.offgrid_marginals(
        marginals=marginals,
        posterior=posterior,
        posterior_previous=posterior_previous,
        t=t,
        t0=t0,
        t1=t1,
        output_scale=output_scale,
    )


def _offgrid_marginals2(*, solution, t, solution_previous, solver):
    return solver.strategy.offgrid_marginals(
        marginals=solution.marginals,
        posterior=solution.posterior,
        posterior_previous=solution_previous.posterior,
        t=t,
        t0=solution_previous.t,
        t1=solution.t,
        output_scale=solution.output_scale,
    )


def log_marginal_likelihood_terminal_values(u, /, *, standard_deviation, posterior):
    """Compute the log-marginal-likelihood at the terminal value.

    Parameters
    ----------
    u
        Observation. Expected to have shape (d,) for an ODE with shape (d,).
    standard_deviation
        Standard deviation of the observation. Expected to be a scalar.
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    """
    if jnp.shape(standard_deviation) != ():
        raise ValueError(
            "Scalar observation noise expected. "
            f"Shape {jnp.shape(standard_deviation)} received."
        )

    # not valid for scalar or matrix-valued solutions
    if jnp.ndim(u) > jnp.ndim(impl.prototypes.qoi()):
        raise ValueError(
            "Terminal-value solution (ndim=1, shape=(n,)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    # Generate an observation-model for the QOI
    model = impl.ssm_util.conditional_to_derivative(0, standard_deviation)
    rv = posterior.init if isinstance(posterior, markov.MarkovSeqRev) else posterior

    _corrected, logpdf = _condition_and_logpdf(rv, u, model)
    return logpdf


def _condition_and_logpdf(rv, data, model):
    observed, conditional = impl.conditional.revert(rv, model)
    corrected = impl.conditional.apply(data, conditional)
    logpdf = impl.stats.logpdf(data, observed)
    return corrected, logpdf


def log_marginal_likelihood(u, /, *, standard_deviation, posterior):
    """Compute the log-marginal-likelihood of observations of the IVP solution.

    Parameters
    ----------
    standard_deviation
        Standard deviation of the observation. Expected to be have shape (n,).
    u
        Observation. Expected to have shape (n, d) for an ODE with shape (d,).
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    strategy
        Strategy (that has been used to compute the solution).
        Expected to correspond to a solution of an ODE with shape (d,).

    !!! note
        Use `log_marginal_likelihood_terminal_values`
        to compute the log-likelihood at the terminal values.

    """
    # todo: complain if it is used with a filter, not a smoother?
    # todo: allow option for log-posterior

    if jnp.shape(standard_deviation) != jnp.shape(u)[:1]:
        raise ValueError(
            f"Observation-noise shape {jnp.shape(standard_deviation)} does not match "
            f"the observation shape {jnp.shape(u)}. "
            f"Expected observation-noise shape: "
            f"{(jnp.shape(u)[0],)} != {jnp.shape(standard_deviation)}. "
        )

    if jnp.ndim(u) < jnp.ndim(impl.prototypes.qoi()) + 1:
        raise ValueError(
            "Time-series solution (ndim=2, shape=(n, m)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    if not isinstance(posterior, markov.MarkovSeqRev):
        msg1 = "Time-series marginal likelihoods "
        msg2 = "cannot be computed with a filtering solution."
        raise TypeError(msg1 + msg2)

    # Generate an observation-model for the QOI
    model_fun = jax.vmap(impl.ssm_util.conditional_to_derivative, in_axes=(None, 0))
    models = model_fun(0, standard_deviation)

    # Select the terminal variable
    rv = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior.init)

    # Run the reverse Kalman filter
    (_corrected, _num_data, logpdf), _ = discrete.kalmanfilter_reverse(
        u, init=rv, conditional=posterior.conditional, observation_model=models
    )

    # Return only the logpdf
    return logpdf
