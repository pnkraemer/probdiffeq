"""Interact with IVP solutions.

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

from probdiffeq.backend import containers, control_flow, functools, random, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any
from probdiffeq.util import filter_util

# TODO: the functions in here should only depend on posteriors / strategies!


class MarkovSeq(containers.NamedTuple):
    """Markov sequence."""

    init: Any
    conditional: Any


def markov_sample(key, markov_seq: MarkovSeq, *, reverse, ssm, shape=()):
    """Sample from a Markov sequence."""
    _assert_filtering_solution_removed(markov_seq)
    # A smoother samples on the grid by sampling i.i.d values
    # from the terminal RV x_N and the backward noises z_(1:N)
    # and then combining them backwards as
    # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
    markov_seq_shape = _sample_shape(markov_seq, ssm=ssm)
    base_samples = random.normal(key, shape=shape + markov_seq_shape)
    return _transform_unit_sample(markov_seq, base_samples, reverse=reverse, ssm=ssm)


def _sample_shape(markov_seq, *, ssm):
    # The number of samples is one larger than the number of conditionals
    _, noise = markov_seq.conditional
    n, *shape_single_sample = ssm.stats.hidden_shape(noise)
    return n + 1, *tuple(shape_single_sample)


def _transform_unit_sample(markov_seq, base_sample, /, reverse, *, ssm):
    if base_sample.ndim > len(_sample_shape(markov_seq, ssm=ssm)):
        transform = functools.partial(_transform_unit_sample, ssm=ssm)
        transform_vmap = functools.vmap(transform, in_axes=(None, 0, None))
        return transform_vmap(markov_seq, base_sample, reverse)

    # Compute a single unit sample.

    def body_fun(samp_prev, conditionals_and_base_samples):
        conditional, base = conditionals_and_base_samples

        rv = ssm.conditional.apply(samp_prev, conditional)
        smp = ssm.stats.transform_unit_sample(base, rv)
        qoi = ssm.stats.qoi_from_sample(smp)
        return smp, qoi

    base_sample_init, base_sample_body = base_sample[0], base_sample[1:]

    # Compute a sample at the terminal value
    init_sample = ssm.stats.transform_unit_sample(base_sample_init, markov_seq.init)
    init_qoi = ssm.stats.qoi_from_sample(init_sample)
    init_val = init_sample

    # Loop over backward models and the remaining base samples
    xs = (markov_seq.conditional, base_sample_body)
    _, qois = control_flow.scan(body_fun, init=init_val, xs=xs, reverse=reverse)
    return qois, init_qoi


def markov_select_terminal(markov_seq: MarkovSeq) -> MarkovSeq:
    """Discard all intermediate filtering solutions from a Markov sequence.

    This function is useful to convert a smoothing-solution into a Markov sequence
    that is compatible with sampling or marginalisation.
    """
    init = tree_util.tree_map(lambda x: x[-1, ...], markov_seq.init)
    return MarkovSeq(init, markov_seq.conditional)


def markov_marginals(markov_seq: MarkovSeq, *, reverse, ssm):
    """Extract the (time-)marginals from a Markov sequence."""
    _assert_filtering_solution_removed(markov_seq)

    def step(x, cond):
        extrapolated = ssm.conditional.marginalise(x, cond)
        return extrapolated, extrapolated

    init, xs = markov_seq.init, markov_seq.conditional
    _, marg = control_flow.scan(step, init=init, xs=xs, reverse=reverse)
    return marg


def _assert_filtering_solution_removed(markov_seq):
    if markov_seq.init.mean.ndim == markov_seq.conditional.noise.mean.ndim:
        msg = "Did you forget to call markov_select_terminal() before?"
        raise ValueError(msg)


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
    offgrid_marginals_vmap = functools.vmap(_offgrid_marginals, in_axes=(0, None, None))
    return offgrid_marginals_vmap(ts, solution, solver)


def _offgrid_marginals(t, solution, solver):
    # side="left" and side="right" are equivalent
    # because we _assume_ that the point sets are disjoint.
    index = np.searchsorted(solution.t, t)

    def _extract_previous(tree):
        return tree_util.tree_map(lambda s: s[index - 1, ...], tree)

    def _extract(tree):
        return tree_util.tree_map(lambda s: s[index, ...], tree)

    marginals = _extract(solution.marginals)

    # In the smoothing context:
    # Extract the correct posterior.init (aka the filtering solutions)
    # The conditionals are incorrect, but we don't really care about this.
    posterior_previous = _extract_previous(solution.posterior)

    t0 = _extract_previous(solution.t)
    t1 = _extract(solution.t)
    output_scale = _extract(solution.output_scale)

    return solver.offgrid_marginals(
        marginals_t1=marginals,
        posterior_t0=posterior_previous,
        t=t,
        t0=t0,
        t1=t1,
        output_scale=output_scale,
    )


def log_marginal_likelihood_terminal_values(
    u, /, *, standard_deviation, posterior, ssm
):
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
    if np.shape(standard_deviation) != ():
        msg = (
            f"Scalar observation noise expected. "
            f"Shape {np.shape(standard_deviation)} received."
        )
        raise ValueError(msg)

    # not valid for scalar or matrix-valued solutions
    if np.ndim(u) > np.ndim(ssm.prototypes.qoi()):
        msg = (
            f"Terminal-value solution (ndim=1, shape=(n,)) expected. "
            f"ndim={np.ndim(u)}, shape={np.shape(u)} received."
        )
        raise ValueError(msg)

    # Generate an observation-model for the QOI
    model = ssm.conditional.to_derivative(0, standard_deviation)
    rv = posterior.init if isinstance(posterior, MarkovSeq) else posterior

    _corrected, logpdf = _condition_and_logpdf(rv, u, model, ssm=ssm)
    return logpdf


def _condition_and_logpdf(rv, data, model, *, ssm):
    observed, conditional = ssm.conditional.revert(rv, model)
    corrected = ssm.conditional.apply(data, conditional)
    logpdf = ssm.stats.logpdf(data, observed)
    return corrected, logpdf


def log_marginal_likelihood(u, /, *, standard_deviation, posterior, ssm):
    """Compute the log-marginal-likelihood of observations of the IVP solution.

    !!! note
        Use `log_marginal_likelihood_terminal_values`
        to compute the log-likelihood at the terminal values.

    Parameters
    ----------
    standard_deviation
        Standard deviation of the observation. Expected to be have shape (n,).
    u
        Observation. Expected to have shape (n, d) for an ODE with shape (d,).
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    """
    # TODO: complain if it is used with a filter, not a smoother?
    # TODO: allow option for log-posterior

    if np.shape(standard_deviation) != np.shape(u)[:1]:
        msg = (
            f"Observation-noise shape {np.shape(standard_deviation)} "
            f"does not match the observation shape {np.shape(u)}. "
            f"Expected observation-noise shape: "
            f"{(np.shape(u)[0],)} != {np.shape(standard_deviation)}. "
        )
        raise ValueError(msg)

    if np.ndim(u) < np.ndim(ssm.prototypes.qoi()) + 1:
        msg = (
            f"Time-series solution (ndim=2, shape=(n, m)) expected. "
            f"ndim={np.ndim(u)}, shape={np.shape(u)} received."
        )
        raise ValueError(msg)

    if not isinstance(posterior, MarkovSeq):
        msg1 = "Time-series marginal likelihoods "
        msg2 = "cannot be computed with a filtering solution."
        raise TypeError(msg1 + msg2)

    # Generate an observation-model for the QOI
    model_fun = functools.vmap(ssm.conditional.to_derivative, in_axes=(None, 0))
    models = model_fun(0, standard_deviation)

    # Select the terminal variable
    rv = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)

    # Run the reverse Kalman filter
    estimator = filter_util.kalmanfilter_with_marginal_likelihood(ssm=ssm)
    (_corrected, _num_data, logpdf), _ = filter_util.estimate_rev(
        u,
        init=rv,
        prior_transitions=posterior.conditional,
        observation_model=models,
        estimator=estimator,
    )

    # Return only the logpdf
    return logpdf


def calibrate(x, /, output_scale, *, ssm):
    """Calibrated a posterior distribution of an IVP solution."""
    if np.ndim(output_scale) > np.ndim(ssm.prototypes.output_scale()):
        output_scale = output_scale[-1]
    if isinstance(x, MarkovSeq):
        return _markov_rescale_cholesky(x, output_scale, ssm=ssm)
    return ssm.stats.rescale_cholesky(x, output_scale)


def _markov_rescale_cholesky(markov_seq: MarkovSeq, factor, *, ssm) -> MarkovSeq:
    """Rescale the Cholesky factor of the covariance of a Markov sequence."""
    init = ssm.stats.rescale_cholesky(markov_seq.init, factor)
    cond = _rescale_cholesky_conditional(markov_seq.conditional, factor, ssm=ssm)
    return MarkovSeq(init=init, conditional=cond)


def _rescale_cholesky_conditional(conditional, factor, /, *, ssm):
    noise_new = ssm.stats.rescale_cholesky(conditional.noise, factor)
    return ssm.conditional.conditional(conditional.matmul, noise_new)
