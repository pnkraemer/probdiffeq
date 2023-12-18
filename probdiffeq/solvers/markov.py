"""Markov sequences and Markov processes."""

from probdiffeq.backend import containers, control_flow, functools, random, tree_util
from probdiffeq.backend.typing import Any
from probdiffeq.impl import impl
from probdiffeq.util import cond_util


class MarkovSeq(containers.NamedTuple):
    """Markov sequence."""

    init: Any
    conditional: Any


def sample(key, markov_seq: MarkovSeq, *, shape, reverse):
    """Sample from a Markov sequence."""
    _assert_filtering_solution_removed(markov_seq)
    # A smoother samples on the grid by sampling i.i.d values
    # from the terminal RV x_N and the backward noises z_(1:N)
    # and then combining them backwards as
    # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
    markov_seq_shape = _sample_shape(markov_seq)
    base_samples = random.normal(key, shape=shape + markov_seq_shape)
    return _transform_unit_sample(markov_seq, base_samples, reverse=reverse)


def _sample_shape(markov_seq):
    # The number of samples is one larger than the number of conditionals
    _, noise = markov_seq.conditional
    n, *shape_single_sample = impl.stats.sample_shape(noise)
    return n + 1, *tuple(shape_single_sample)


def _transform_unit_sample(markov_seq, base_sample, /, reverse):
    if base_sample.ndim > len(_sample_shape(markov_seq)):
        transform_vmap = functools.vmap(_transform_unit_sample, in_axes=(None, 0, None))
        return transform_vmap(markov_seq, base_sample, reverse)

    # Compute a single unit sample.

    def body_fun(carry, conditionals_and_base_samples):
        _, samp_prev = carry
        conditional, base = conditionals_and_base_samples

        rv = impl.conditional.apply(samp_prev, conditional)
        smp = impl.variable.transform_unit_sample(base, rv)
        qoi = impl.hidden_model.qoi_from_sample(smp)
        return (qoi, smp), (qoi, smp)

    base_sample_init, base_sample_body = base_sample[0], base_sample[1:]

    # Compute a sample at the terminal value
    init_sample = impl.variable.transform_unit_sample(base_sample_init, markov_seq.init)
    init_qoi = impl.hidden_model.qoi_from_sample(init_sample)
    init_val = (init_qoi, init_sample)

    # Loop over backward models and the remaining base samples
    xs = (markov_seq.conditional, base_sample_body)
    _, (qois, samples) = control_flow.scan(
        body_fun, init=init_val, xs=xs, reverse=reverse
    )
    return (qois, samples), (init_qoi, init_sample)


def rescale_cholesky(markov_seq: MarkovSeq, factor) -> MarkovSeq:
    """Rescale the Cholesky factor of the covariance of a Markov sequence."""
    init = impl.variable.rescale_cholesky(markov_seq.init, factor)
    cond = _rescale_cholesky_conditional(markov_seq.conditional, factor)
    return MarkovSeq(init=init, conditional=cond)


def _rescale_cholesky_conditional(conditional, factor, /):
    noise_new = impl.variable.rescale_cholesky(conditional.noise, factor)
    return cond_util.Conditional(conditional.matmul, noise_new)


def select_terminal(markov_seq: MarkovSeq) -> MarkovSeq:
    """Discard all intermediate filtering solutions from a Markov sequence.

    This function is useful to convert a smoothing-solution into a Markov sequence
    that is compatible with sampling or marginalisation.
    """
    init = tree_util.tree_map(lambda x: x[-1, ...], markov_seq.init)
    return MarkovSeq(init, markov_seq.conditional)


def marginals(markov_seq: MarkovSeq, *, reverse):
    """Extract the (time-)marginals from a Markov sequence."""
    _assert_filtering_solution_removed(markov_seq)

    def step(x, cond):
        extrapolated = impl.conditional.marginalise(x, cond)
        return extrapolated, extrapolated

    init, xs = markov_seq.init, markov_seq.conditional
    _, marg = control_flow.scan(step, init=init, xs=xs, reverse=reverse)
    return marg


def _assert_filtering_solution_removed(markov_seq):
    if markov_seq.init.mean.ndim == markov_seq.conditional.noise.mean.ndim:
        msg = "Did you forget to call markov.select_terminal() before?"
        raise ValueError(msg)
