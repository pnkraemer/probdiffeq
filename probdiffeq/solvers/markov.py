"""Markov sequences and Markov processes."""

from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers
from probdiffeq.impl import impl


class MarkovSeqRev(containers.NamedTuple):
    init: Any
    conditional: Any


def sample(key, markov_seq: MarkovSeqRev, *, shape):
    # A smoother samples on the grid by sampling i.i.d values
    # from the terminal RV x_N and the backward noises z_(1:N)
    # and then combining them backwards as
    # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
    markov_seq_shape = _sample_shape(markov_seq)
    base_samples = jax.random.normal(key=key, shape=shape + markov_seq_shape)
    return _transform_unit_sample(markov_seq, base_samples)


def _sample_shape(markov_seq):
    # The number of samples is one larger than the number of conditionals
    _, noise = markov_seq.conditional
    n, *shape_single_sample = impl.stats.sample_shape(noise)
    return n + 1, *tuple(shape_single_sample)


def _transform_unit_sample(markov_seq, base_sample, /):
    if base_sample.ndim > len(_sample_shape(markov_seq)):
        transform_vmap = jax.vmap(_transform_unit_sample, in_axes=(None, 0))
        return transform_vmap(markov_seq, base_sample)

    # Compute a single unit sample.

    def body_fun(carry, conditionals_and_base_samples):
        _, samp_prev = carry
        conditional, base = conditionals_and_base_samples

        rv = impl.conditional.apply(samp_prev, conditional)
        sample = impl.variable.transform_unit_sample(base, rv)
        qoi = impl.hidden_model.qoi_from_sample(sample)
        return (qoi, sample), (qoi, sample)

    # Compute a sample at the terminal value
    init = jax.tree_util.tree_map(lambda s: s[-1, ...], markov_seq.init)
    init_sample = impl.variable.transform_unit_sample(base_sample[-1], init)
    init_qoi = impl.hidden_model.qoi_from_sample(init_sample)
    init_val = (init_qoi, init_sample)

    # Loop over backward models and the remaining base samples
    xs = (markov_seq.conditional, base_sample[:-1])
    _, (qois, samples) = jax.lax.scan(f=body_fun, init=init_val, xs=xs, reverse=True)
    qois_full = jnp.concatenate((qois, init_qoi[None, ...]))
    samples_full = jnp.concatenate((samples, init_sample[None, ...]))
    return qois_full, samples_full


def rescale_cholesky(markov_seq: MarkovSeqRev, factor) -> MarkovSeqRev:
    init = impl.variable.rescale_cholesky(markov_seq.init, factor)
    A, noise = markov_seq.conditional
    noise = impl.variable.rescale_cholesky(noise, factor)
    return MarkovSeqRev(init=init, conditional=(A, noise))


def marginals(markov_seq: MarkovSeqRev):
    def step(x, cond):
        extrapolated = impl.conditional.marginalise(x, cond)
        return extrapolated, extrapolated

    # If we hold many 'init's, choose the terminal one.
    # todo: should we let the user do this?
    _, noise = markov_seq.conditional
    if noise.mean.shape == markov_seq.init.mean.shape:
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], markov_seq.init)
    else:
        init = markov_seq.init

    _, marg = jax.lax.scan(step, init=init, xs=markov_seq.conditional, reverse=True)
    return marg
