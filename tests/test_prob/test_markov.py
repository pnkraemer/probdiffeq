"""Tests for Markov process machinery."""

import jax.numpy as jnp
import pytest_cases

from odefilter.prob import markov, rv


@pytest_cases.case
def case_univariate():
    ones = jnp.ones(10)
    zeros = jnp.zeros(10)

    return markov.MarkovSequence(
        transition_operators=ones,
        transition_noise_rvs=rv.Normal(mean=zeros, cov_sqrtm_upper=ones),
        init=rv.Normal(mean=0.0, cov_sqrtm_upper=1.0),
    )


@pytest_cases.case
def case_multivariate():
    eyes = jnp.stack([jnp.eye(2)] * 10)
    zeros = jnp.zeros((10, 2))

    return markov.MarkovSequence(
        transition_operators=eyes,
        transition_noise_rvs=rv.Normal(mean=zeros, cov_sqrtm_upper=eyes),
        init=rv.Normal(mean=zeros[0], cov_sqrtm_upper=eyes[0]),
    )


@pytest_cases.parametrize_with_cases("markov_sequence", cases=".")
def test_marginalise_sequence(markov_sequence):

    rvs = markov.marginalise_sequence(markov_sequence=markov_sequence)

    assert isinstance(rvs, type(markov_sequence.transition_noise_rvs))
    assert rvs[0].shape == markov_sequence.transition_noise_rvs[0].shape
    assert rvs[1].shape == markov_sequence.transition_noise_rvs[1].shape
