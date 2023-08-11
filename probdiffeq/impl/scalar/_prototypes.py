import jax.numpy as jnp

from probdiffeq.impl import _prototypes
from probdiffeq.impl.scalar import _normal


class PrototypeBackend(_prototypes.PrototypeBackend):
    def qoi(self):
        return jnp.empty(())

    def observed(self):
        mean = jnp.empty(())
        cholesky = jnp.empty(())
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return jnp.empty(())

    def output_scale(self):
        return jnp.empty(())
