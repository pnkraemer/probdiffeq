"""Random variable utilities."""

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float


class Normal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Any
    cov_sqrtm_upper: Any


class IsotropicNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, "n d"]
    cov_sqrtm_upper: Float[Array, "n n"]
