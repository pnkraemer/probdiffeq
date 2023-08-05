from typing import Any

import jax.numpy as jnp

from probdiffeq.backend import containers
from probdiffeq.statespace.backend import rv


class Normal(containers.NamedTuple):
    mean: Any
    cholesky: Any


class RandomVariableBackEnd(rv.RandomVariableBackEnd):
    def qoi_like(self):
        mean = jnp.empty(())
        cholesky = jnp.empty(())
        return Normal(mean, cholesky)
