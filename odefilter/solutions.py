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


class SmoothingPosterior(Generic[NormalLike], eqx.Module):
    """Markov sequences as smoothing solutions."""

    filtered: NormalLike
    backward_model: BackwardModel[NormalLike]


# todo: move to implementations?


def marginalise_sequence_isotropic(*, init, backward_model):
    """Compute marginals of a markov sequence."""

    def body_fun(carry, x):
        linop, noise = x.transition, x.noise
        out = marginalise_model_isotropic(init=carry, linop=linop, noise=noise)
        return out, out

    _, rvs = jax.lax.scan(f=body_fun, init=init, xs=backward_model, reverse=False)
    return rvs


def marginalise_model_isotropic(*, init, linop, noise):
    """Marginalise the output of a linear model."""
    # Apply transition
    m_new = jnp.dot(linop, init.mean) + noise.mean
    l_new = sqrtm.sum_of_sqrtm_factors(
        R1=jnp.dot(linop, init.cov_sqrtm_lower).T, R2=noise.cov_sqrtm_lower.T
    ).T

    return IsotropicNormal(mean=m_new, cov_sqrtm_lower=l_new)
