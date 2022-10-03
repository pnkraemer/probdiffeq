"""Markov process, Markov sequences and Markov machinery."""

from typing import Any, Callable, Generic, NamedTuple, TypeVar

import jax.lax
import jax.numpy as jnp

from odefilter import rv, sqrtm


class MarkovSequence(NamedTuple):
    """Markov sequence.

    Markov sequences are discretised Markov processes.

    .. math::
        ABC = DEF
    """

    transition_operators: Any
    """Transition operators. Matrix or linear operator."""

    transition_noise_rvs: rv.Normal
    """Process noises."""

    init: rv.Normal
    """Initial random variable."""

    reverse: bool = False
    """Temporal direction of the transitions."""

    @property
    def transitions(self):
        """Group transition operators and process noises.

        Used to simplify the call to :func:`jax.lax.scan`.
        """
        return self.transition_operators, self.transition_noise_rvs


def marginalise_sequence(*, markov_sequence):
    """Compute marginals of a markov sequence."""

    def body_fun(carry, x):
        op, noise = x
        out = marginalise(init=carry, operator=op, noise=noise)
        return out, out

    _, rvs = jax.lax.scan(
        f=body_fun,
        init=markov_sequence.init,
        xs=markov_sequence.transitions,
        reverse=markov_sequence.reverse,
    )
    return rvs


def marginalise(*, init, operator, noise):
    """Marginalise the output of a linear model."""
    # Read system matrices
    mean, cov_sqrtm_upper = init
    noise_mean, noise_cov_sqrtm_upper = noise

    # Apply transition
    mean_new = jnp.dot(operator, mean) + noise_mean
    cov_sqrtm_upper_new = sqrtm.sum_of_sqrtm_factors(
        R1=jnp.dot(operator, cov_sqrtm_upper), R2=noise_cov_sqrtm_upper
    )

    # Output type equals input type.
    rv_new = init._replace(mean=mean_new, cov_sqrtm_upper=cov_sqrtm_upper_new)
    return rv_new
