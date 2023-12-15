"""Conditional-utilities."""
from typing import Any

from probdiffeq.backend import containers
from probdiffeq.backend.typing import Array

# One class to unify the outputs of all the conditional/transform machinery.


class Conditional(containers.NamedTuple):
    """Conditional distributions."""

    matmul: Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type
