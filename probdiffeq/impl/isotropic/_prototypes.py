import jax.numpy as jnp

from probdiffeq.impl import _prototypes
from probdiffeq.impl.isotropic import _normal


class PrototypeBackend(_prototypes.PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self):
        return jnp.empty(self.ode_shape)

    def observed(self):
        mean = jnp.empty((1, *self.ode_shape))
        cholesky = jnp.empty(())
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return jnp.empty(())

    def output_scale(self):
        return jnp.empty(())
