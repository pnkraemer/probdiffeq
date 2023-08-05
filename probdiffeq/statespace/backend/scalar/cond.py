import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.backend import cond
from probdiffeq.statespace.backend.scalar import rv


class ConditionalBackEnd(cond.ConditionalBackEnd):
    def marginalise_transformation(self, x: rv.Normal, transformation, /):
        A, b = transformation
        mean, cov_sqrtm_lower = x.mean, x.cov_sqrtm_lower

        cov_sqrtm_lower_new = _sqrt_util.triu_via_qr(A(cov_sqrtm_lower)[:, None])
        cov_sqrtm_lower_squeezed = jnp.reshape(cov_sqrtm_lower_new, ())
        return rv.Normal(A(mean) + b, cov_sqrtm_lower_squeezed)
