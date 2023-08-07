import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _conditional
from probdiffeq.statespace.blockdiag import _normal


class ConditionalBackend(_conditional.ConditionalBackend):
    def apply(self, x, conditional, /):
        matrix, noise = conditional
        assert matrix.ndim == 3
        mean = jnp.einsum("ijk,ik->ij", matrix, x) + noise.mean
        return _normal.Normal(mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional
        assert matrix.ndim == 3

        mean = jnp.einsum("ijk,ik->ij", matrix, rv.mean) + noise.mean

        chol1 = _transpose(jnp.einsum("ijk,ikk->ijk", matrix, rv.cholesky))
        chol2 = _transpose(noise.cholesky)
        R_stack = (chol1, chol2)
        cholesky = jax.vmap(_sqrt_util.sum_of_sqrtm_factors)(R_stack)
        return _normal.Normal(mean, _transpose(cholesky))

    def merge(self, cond1, cond2, /):
        raise NotImplementedError

    def revert(self, rv, conditional, /):
        raise NotImplementedError


def _transpose(matrix):
    return jnp.transpose(matrix, axes=(0, -1, -2))
