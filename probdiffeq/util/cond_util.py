"""Conditional-utilities."""
from probdiffeq.backend import containers
from probdiffeq.backend.typing import Any, Array

# One class to unify the outputs of all the conditional/transform machinery.


class Conditional(containers.NamedTuple):
    """Conditional distributions."""

    matmul: Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type
