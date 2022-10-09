"""Tests for Markov process machinery."""

import jax.numpy as jnp
import pytest_cases

from odefilter import markov, solution


@pytest_cases.case
def case_univariate():
    ones = jnp.ones(10)
    zeros = jnp.zeros(10)

    return markov.MarkovSequence(
        transition_operators=ones,
        transition_noise_rvs=solution.Normal(mean=zeros, cov_sqrtm_lower=ones),
        init=solution.Normal(mean=0.0, cov_sqrtm_lower=1.0),
    )


@pytest_cases.case
def case_multivariate():
    eyes = jnp.stack([jnp.eye(2)] * 10)
    zeros = jnp.zeros((10, 2))

    return markov.MarkovSequence(
        transition_operators=eyes,
        transition_noise_rvs=solution.Normal(mean=zeros, cov_sqrtm_lower=eyes),
        init=solution.Normal(mean=zeros[0], cov_sqrtm_lower=eyes[0]),
    )


@pytest_cases.parametrize_with_cases("markov_sequence", cases=".")
def test_marginalise_sequence(markov_sequence):

    rvs = markov.marginalise_sequence(markov_sequence=markov_sequence)

    c1shape = rvs.cov_sqrtm_lower.shape
    c2shape = markov_sequence.transition_noise_rvs.cov_sqrtm_lower.shape
    assert isinstance(rvs, type(markov_sequence.transition_noise_rvs))
    assert rvs.mean.shape == markov_sequence.transition_noise_rvs.mean.shape
    assert c1shape == c2shape
