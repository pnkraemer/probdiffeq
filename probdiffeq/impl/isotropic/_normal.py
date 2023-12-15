from probdiffeq.backend import containers
from probdiffeq.backend.typing import Array


class Normal(containers.NamedTuple):
    mean: Array
    cholesky: Array
