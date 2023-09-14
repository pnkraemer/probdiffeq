import jax.numpy as jnp

from probdiffeq.impl import _variable
from probdiffeq.impl.isotropic import _normal
from probdiffeq.impl.util import cond_util


class VariableBackend(_variable.VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def rescale_cholesky_conditional(self, conditional, factor, /):
        noise_new = self.rescale_cholesky(conditional.noise, factor)
        return cond_util.Conditional(conditional.matmul, noise_new)

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, rv):
        eye_d = jnp.eye(*self.ode_shape)
        cov = rv.cholesky @ rv.cholesky.T
        cov = jnp.kron(eye_d, cov)
        mean = rv.mean.reshape((-1,), order="F")
        return (mean, cov)
