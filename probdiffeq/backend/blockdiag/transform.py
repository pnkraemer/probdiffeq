import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.backend import _transform
from probdiffeq.backend.blockdiag import _normal


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        cov_new = jax.vmap(_sqrt_util.triu_via_qr)(A(cholesky)[:, :, None])
        cov_new = jnp.squeeze(cov_new, axis=(-2, -1))
        return _normal.Normal(A(mean) + b, cov_new)

    def revert(self, rv, transformation, /):
        raise NotImplementedError
