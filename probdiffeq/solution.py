"""Interact with estimated solutions (on dense grids).

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _markov
from probdiffeq.backend import containers
from probdiffeq.impl import impl

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


def log_marginal_likelihood_terminal_values(*, observation_std, u, posterior, strategy):
    """Compute the log-marginal-likelihood of \
     observations of the IVP solution at the terminal value.

    Parameters
    ----------
    observation_std
        Standard deviation of the observation. Expected to be a scalar.
    u
        Observation. Expected to have shape (d,) for an ODE with shape (d,).
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    strategy
        Strategy (that has been used to compute the solution).
        Expected to correspond to a solution of an ODE with shape (d,).
    """
    if jnp.shape(observation_std) != ():
        raise ValueError(
            "Scalar observation noise expected. "
            f"Shape {jnp.shape(observation_std)} received."
        )

    if jnp.ndim(u) >= 2:  # not valid for scalar or matrix-valued solutions
        raise ValueError(
            "Terminal-value solution (ndim=1, shape=(n,)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    # Generate an observation-model for the QOI
    model = impl.ssm_util.conditional_to_derivative(0, observation_std)
    if isinstance(posterior, _markov.MarkovSeqRev):
        rv = posterior.init
    else:
        rv = posterior
    init = _initialise(rv, u, model)
    return init.logpdf


def log_marginal_likelihood(*, observation_std, u, posterior, strategy):
    """Compute the log-marginal-likelihood of \
     observations of the IVP solution.

    Parameters
    ----------
    observation_std
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

    if jnp.shape(observation_std) != (jnp.shape(u)[0],):
        raise ValueError(
            f"Observation-noise shape {jnp.shape(observation_std)} does not match "
            f"the observation shape {jnp.shape(u)}. "
            f"Expected observation-noise shape: "
            f"{(jnp.shape(u)[0],)} != {jnp.shape(observation_std)}. "
        )

    if jnp.ndim(u) < 2:
        raise ValueError(
            "Time-series solution (ndim=2, shape=(n, m)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    if not isinstance(posterior, _markov.MarkovSeqRev):
        msg1 = "Time-series marginal likelihoods "
        msg2 = "cannot be computed with a filtering solution."
        raise TypeError(msg1 + msg2)

    result = _kalman_filter(u, posterior, observation_std, strategy=strategy)
    return result


# todo: this smells a lot like a `impl.SSV` object.
#  But merging those two data structures might be in the far future.


class _KalmanFilterState(containers.NamedTuple):
    rv: Any
    num_data_points: int
    logpdf: float


# todo: this should return a Filtering posterior or a smoothing posterior
#  which could then be plotted. Right?
#  (We might also want some dense-output/checkpoint kind of thing here)
# todo: we should reuse the extrapolation model impl.
#  But this only works if the ODE posterior uses the preconditioner (I think).
# todo: we should allow proper noise, and proper information functions.
#  But it is not clear which data structure that should be.
def _kalman_filter(u, /, mseq, standard_deviations, *, strategy, reverse=True):
    # Generate an observation-model for the QOI
    model_fun = jax.vmap(impl.ssm_util.conditional_to_derivative, in_axes=(None, 0))
    models = model_fun(0, standard_deviations)

    # Incorporate final data point

    def select(tree, idx_or_slice):
        return jax.tree_util.tree_map(lambda s: s[idx_or_slice, ...], tree)

    rv, data, model = select(tree=(mseq.init, u, models), idx_or_slice=-1)
    init = _initialise(rv, data, model)

    # Scan over the remaining data points
    data, observation = select((u, models), idx_or_slice=slice(0, -1, 1))
    kalman_state, _ = jax.lax.scan(
        f=lambda *a: (_step(*a), None),
        init=init,
        xs=(data, mseq.conditional, observation),
        reverse=reverse,
    )
    return kalman_state.logpdf


def _initialise(rv, data, model):
    observed, conditional = impl.conditional.revert(rv, model)
    corrected = impl.conditional.apply(data, conditional)
    logpdf = impl.random.logpdf(data, observed)
    return _KalmanFilterState(corrected, 1.0, logpdf)


def _step(state, problem):
    data, conditional, observation = problem
    rv, num_data, logpdf = state

    rv = impl.conditional.marginalise(rv, conditional)
    observed, reverse = impl.conditional.revert(rv, observation)
    corrected = impl.conditional.apply(data, reverse)

    logpdf_new = impl.random.logpdf(data, observed)
    logpdf_mean = impl.ssm_util.update_mean(logpdf, logpdf_new, num_data)
    return _KalmanFilterState(corrected, num_data + 1.0, logpdf_mean)
