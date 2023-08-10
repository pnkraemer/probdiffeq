"""Random-variable transformation."""
import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _cond_util, _transform
from probdiffeq.impl.blockdiag import _normal


class TransformBackend(_transform.TransformBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        A_cholesky = A @ cholesky
        cholesky = jax.vmap(_sqrt_util.triu_via_qr)(_transpose(A_cholesky))
        mean = A @ mean + b
        return _normal.Normal(mean, cholesky)

    def revert(self, rv, transformation, /):
        A, bias = transformation
        cholesky_upper = jnp.transpose(rv.cholesky, axes=(0, -1, -2))
        A_cholesky_upper = _transpose(A @ rv.cholesky)

        revert_fun = jax.vmap(_sqrt_util.revert_conditional_noisefree)
        r_obs, (r_cor, gain) = revert_fun(A_cholesky_upper, cholesky_upper)
        cholesky_obs = _transpose(r_obs)
        cholesky_cor = _transpose(r_cor)

        # Gather terms and return
        mean_observed = (A @ rv.mean) + bias
        m_cor = rv.mean - (gain * (mean_observed[..., None]))[..., 0]
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, _cond_util.Conditional(gain, corrected)


def _transpose(arr, /):
    return jnp.transpose(arr, axes=(0, 2, 1))