"""Conditional-utilities."""
from typing import Any

import jax

from probdiffeq.backend import containers
from probdiffeq.impl import impl

# One class to unify the outputs of all the conditional/transform machinery.


class Conditional(containers.NamedTuple):
    matmul: jax.Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type


def rescale_cholesky_conditional(conditional, factor, /):
    noise_new = impl.variable.rescale_cholesky(conditional.noise, factor)
    return Conditional(conditional.matmul, noise_new)
