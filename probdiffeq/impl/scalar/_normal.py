from typing import Any

from probdiffeq.backend import containers


class Normal(containers.NamedTuple):
    mean: Any
    cholesky: Any
