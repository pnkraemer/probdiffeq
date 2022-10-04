"""Random variable utilities."""

from typing import Any, NamedTuple


class Normal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Any
    cov_sqrtm_upper: Any
