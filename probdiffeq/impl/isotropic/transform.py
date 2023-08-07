import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _transform
from probdiffeq.impl.isotropic import _normal


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky
        cholesky_new = _sqrt_util.triu_via_qr((A @ cholesky)[None, ...].T)
        cholesky_squeezed = jnp.reshape(cholesky_new, ())
        return _normal.Normal((A @ mean) + b, cholesky_squeezed)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky)[None, ...].T, R_X=cholesky.T
        )
        cholesky_obs = jnp.reshape(r_obs, ())
        cholesky_cor = r_cor.T

        # Gather terms and return
        mean_observed = A @ mean + b
        m_cor = mean - gain * (mean_observed[None, ...])
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, (corrected, gain)
