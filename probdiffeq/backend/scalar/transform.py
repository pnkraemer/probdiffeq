import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.backend import _transform
from probdiffeq.backend.scalar import _normal


class TransformBackEnd(_transform.TransformBackEnd):
    def marginalise(self, rv, transformation, /):
        # currently, assumes that A(rv.cholesky) is a vector, not a matrix.
        matmul, b = transformation
        cholesky_new = _sqrt_util.triu_via_qr(matmul(rv.cholesky)[:, None])
        cholesky_new_squeezed = jnp.reshape(cholesky_new, ())
        return _normal.Normal(matmul(rv.mean) + b, cholesky_new_squeezed)

    def revert(self, rv, transformation, /):
        # Assumes that A maps a vector to a scalar...

        # Extract information
        A, b = transformation

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree
        #  to transformation_revert_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=A(rv.cholesky)[:, None], R_X=rv.cholesky.T
        )
        cov_sqrtm_lower_obs = jnp.reshape(r_obs, ())
        cov_sqrtm_lower_cor = r_cor.T
        gain = jnp.squeeze(gain, axis=-1)

        # Gather terms and return
        m_cor = rv.mean - gain * (A(rv.mean) + b)
        corrected = _normal.Normal(m_cor, cov_sqrtm_lower_cor)
        observed = _normal.Normal(A(rv.mean) + b, cov_sqrtm_lower_obs)
        return observed, (corrected, gain)
