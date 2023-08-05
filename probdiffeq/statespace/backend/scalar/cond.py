import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.backend import _cond
from probdiffeq.statespace.backend.scalar import random


class ConditionalBackEnd(_cond.ConditionalBackEnd):
    def marginalise_transformation(self, x, transformation, /):
        A, b = transformation
        mean, cov_sqrtm_lower = x.mean, x.cov_sqrtm_lower

        cov_sqrtm_lower_new = _sqrt_util.triu_via_qr(A(cov_sqrtm_lower)[:, None])
        cov_sqrtm_lower_squeezed = jnp.reshape(cov_sqrtm_lower_new, ())
        return random.Normal(A(mean) + b, cov_sqrtm_lower_squeezed)

    def revert_transformation(self, rv, transformation, /):
        # Extract information
        A, b = transformation
        mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=A(cov_sqrtm_lower)[:, None], R_X=cov_sqrtm_lower.T
        )
        cov_sqrtm_lower_obs = jnp.reshape(r_obs, ())
        cov_sqrtm_lower_cor = r_cor.T
        gain = jnp.squeeze(gain, axis=-1)

        # Gather terms and return
        m_cor = mean - gain * (A(mean) + b)
        corrected = random.Normal(m_cor, cov_sqrtm_lower_cor)
        observed = random.Normal(A(mean) + b, cov_sqrtm_lower_obs)
        return observed, (corrected, gain)
