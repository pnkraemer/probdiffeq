"""Conditional-utilities."""
from typing import Any

import jax

from probdiffeq.backend import containers

# One class to unify the outputs of all the conditional/transform machinery.


class Conditional(containers.NamedTuple):
    """Conditional distributions."""

    matmul: jax.Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type
