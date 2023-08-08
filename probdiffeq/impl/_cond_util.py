from typing import Any

import jax

from probdiffeq.backend import containers


class Conditional(containers.NamedTuple):
    matmul: jax.Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type
