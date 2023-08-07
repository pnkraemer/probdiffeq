import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _transform
from probdiffeq.impl.blockdiag import _normal


class TransformBackend(_transform.TransformBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        cov_new = jax.vmap(_sqrt_util.triu_via_qr)((A @ cholesky)[:, :, None])
        cov_new = jnp.squeeze(cov_new, axis=(-2, -1))
        return _normal.Normal(A @ mean + b, cov_new)

    def revert(self, rv, transformation, /):
        A, bias = transformation

        # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
        cholesky_upper = jnp.transpose(rv.cholesky, axes=(0, -1, -2))
        r_obs, (r_cor, gain) = jax.vmap(_sqrt_util.revert_conditional_noisefree)(
            (A @ rv.cholesky)[..., None], cholesky_upper
        )
        cholesky_obs = jnp.reshape(r_obs, (-1,))
        cholesky_cor = jnp.transpose(r_cor, axes=(0, -1, -2))
        # Gather terms and return
        mean_observed = (A @ rv.mean) + bias
        gain = jnp.reshape(gain, self.ode_shape + (-1,))

        m_cor = rv.mean - (gain * (mean_observed[..., None]))
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, (corrected, gain)
