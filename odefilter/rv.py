"""Random variable utilities."""

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float


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
