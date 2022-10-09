"""ODE filter solutions components.

By construction (extrapolate-correct, not correct-extrapolate)
the solution intervals are right-including, i.e. defined
on the interval $(t_0, t_1]$.
"""


from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from odefilter import sqrtm


class Normal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Any
    cov_sqrtm_lower: Any


class IsotropicNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, "n d"]
    cov_sqrtm_lower: Float[Array, "n n"]


class MultivariateNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, " k"]
    cov_sqrtm_lower: Float[Array, "k k"]


NormalLike = TypeVar("RVLike", Normal, IsotropicNormal)
"""A type-variable to alias appropriate Normal-like random variables."""


class BackwardModel(Generic[NormalLike], eqx.Module):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: NormalLike


class MarkovSequence(Generic[NormalLike], eqx.Module):
    """Markov sequences as smoothing solutions."""

    # by default: reverse!
    # make some noise if you want forward!
    init: NormalLike
    backward_model: BackwardModel[NormalLike]


def marginalise_sequence(*, markov_sequence):
    """Compute marginals of a markov sequence."""

    def body_fun(carry, x):
        op, noise = x.transition, x.noise
        out = marginalise_model(init=carry, operator=op, noise=noise)
        return out, out

    _, rvs = jax.lax.scan(
        f=body_fun,
        init=markov_sequence.init,
        xs=markov_sequence.backward_model,
        reverse=False,
    )
    return rvs


def marginalise_model(*, init, operator, noise):
    """Marginalise the output of a linear model."""
    # Read system matrices
    mean, cov_sqrtm_lower = init.mean, init.cov_sqrtm_lower
    noise_mean, noise_cov_sqrtm_lower = noise.mean, noise.cov_sqrtm_lower

    # Apply transition
    mean_new = jnp.dot(operator, mean) + noise_mean
    cov_sqrtm_lower_new = sqrtm.sum_of_sqrtm_factors(
        R1=jnp.dot(operator, cov_sqrtm_lower).T, R2=noise_cov_sqrtm_lower.T
    ).T

    # Output type equals input type.
    rv_new = Normal(mean=mean_new, cov_sqrtm_lower=cov_sqrtm_lower_new)
    return rv_new
