"""ODE filter solutions components.

By construction (extrapolate-correct, not correct-extrapolate)
the solution intervals are right-including, i.e. defined
on the interval $(t_0, t_1]$.
"""


from typing import Any, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Float


class IsotropicNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, "n d"]
    cov_sqrtm_lower: Float[Array, "n n"]


class MultivariateNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, " k"]
    cov_sqrtm_lower: Float[Array, "k k"]


NormalLike = TypeVar("RVLike", MultivariateNormal, IsotropicNormal)
"""A type-variable to alias appropriate Normal-like random variables."""


class BackwardModel(Generic[NormalLike], eqx.Module):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: NormalLike


class SmoothingPosterior(Generic[NormalLike], eqx.Module):
    """Markov sequences as smoothing solutions."""

    filtered: NormalLike
    backward_model: BackwardModel[NormalLike]
