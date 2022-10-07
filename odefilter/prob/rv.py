"""Random variable utilities."""

from typing import Any

import equinox as eqx


class Normal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Any
    cov_sqrtm_upper: Any
