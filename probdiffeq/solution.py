"""Interact with estimated solutions (on dense grids).

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _markov
from probdiffeq.backend import containers

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

    ssv = strategy.init(None, posterior).ssv
    obs, _ = ssv.observe_qoi(observation_std=observation_std)
    return jnp.sum(obs.logpdf(u))


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


# todo: this smells a lot like a `statespace.SSV` object.
#  But merging those two data structures might be in the far future.


class _KalFiltState(containers.NamedTuple):
    rv: Any
    num_data_points: int
    log_marginal_likelihood: float


# todo: this should return a Filtering posterior or a smoothing posterior
#  which could then be plotted. Right?
#  (We might also want some dense-output/checkpoint kind of thing here)
# todo: we should reuse the extrapolation model statespace.
#  But this only works if the ODE posterior uses the preconditioner (I think).
# todo: we should allow proper noise, and proper information functions.
#  But it is not clear which data structure that should be.
def _kalman_filter(u, /, mseq, standard_deviations, *, strategy, reverse=True):
    # Incorporate final data point
    rv_terminal = jax.tree_util.tree_map(lambda x: x[-1, ...], mseq.init)
    init = _init_fn(rv_terminal, (standard_deviations[-1], u[-1]), strategy=strategy)

    # Scan over the remaining data points
    lml_state, _ = jax.lax.scan(
        f=functools.partial(_filter_step, strategy=strategy),
        init=init,
        xs=(mseq.conditional, (standard_deviations[:-1], u[:-1])),
        reverse=reverse,
    )
    return lml_state.log_marginal_likelihood


def _init_fn(rv, problem, *, strategy):
    obs_std, data = problem

    rv_as_mseq = _markov.MarkovSeqRev(init=rv, conditional=None)
    ssv, _ = strategy.extrapolation.init(rv_as_mseq)
    obs, cond_cor = ssv.observe_qoi(observation_std=obs_std)

    cor = cond_cor(data)
    lml_new = jnp.sum(obs.logpdf(data))
    return _KalFiltState(cor, 1.0, log_marginal_likelihood=lml_new)


def _filter_step(state, problem, *, strategy):
    bw_model, (obs_std, data) = problem

    state = _predict(state, problem=bw_model)
    updates = _update(state, problem=(obs_std, data), strategy=strategy)
    state = _apply_updates(state, updates=updates)
    return state, state


def _predict(state, *, problem):
    """Extrapolate according to the given transition."""
    rv = problem.marginalise(state.rv)
    return _KalFiltState(rv, state.num_data_points, state.log_marginal_likelihood)


def _update(state, problem, *, strategy):
    """Observe the QOI and compute the 'local' log-marginal likelihood."""
    obs_std, data = problem

    rv_as_mseq = _markov.MarkovSeqRev(init=state.rv, conditional=None)
    ssv, _ = strategy.extrapolation.init(rv_as_mseq)
    observed, conditional = ssv.observe_qoi(observation_std=obs_std)

    corrected = conditional(data)
    lml = jnp.sum(observed.logpdf(data))
    return corrected, lml


def _apply_updates(state, /, *, updates):
    """Update the 'global' log-marginal-likelihood and return a new state."""
    corrected, lml_new = updates
    num_data = state.num_data_points
    lml_prev = state.log_marginal_likelihood

    lml_updated = (num_data * lml_prev + lml_new) / (num_data + 1)
    return _KalFiltState(corrected, num_data + 1, log_marginal_likelihood=lml_updated)
