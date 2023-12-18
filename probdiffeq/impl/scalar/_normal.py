from probdiffeq.backend import containers
from probdiffeq.backend.typing import Any


class Normal(containers.NamedTuple):
    mean: Any
    cholesky: Any
