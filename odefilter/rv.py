"""Random variable utilities"""

from typing import Any, Generic, NamedTuple, TypeVar


class Normal(NamedTuple):
    mean: Any
    cov_sqrtm_upper: Any
