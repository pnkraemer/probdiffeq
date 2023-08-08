"""Conditional implementation."""
import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _cond_util, _conditional
from probdiffeq.impl.blockdiag import _normal


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
        A, noise = conditional
        rv_chol_upper = jnp.transpose(rv.cholesky, axes=(0, -1, -2))
        noise_chol_upper = jnp.transpose(noise.cholesky, axes=(0, -1, -2))
        A_rv_chol_upper = jnp.transpose(A @ rv.cholesky, axes=(0, -1, -2))

        revert = jax.vmap(_sqrt_util.revert_conditional)
        r_obs, (r_cor, gain) = revert(A_rv_chol_upper, rv_chol_upper, noise_chol_upper)

        cholesky_obs = jnp.transpose(r_obs, axes=(0, -1, -2))
        cholesky_cor = jnp.transpose(r_cor, axes=(0, -1, -2))

        # Gather terms and return
        mean_observed = (A @ rv.mean[..., None])[..., 0] + noise.mean
        m_cor = rv.mean - (gain @ (mean_observed[..., None]))[..., 0]
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, _cond_util.Conditional(gain, corrected)


def _transpose(matrix):
    return jnp.transpose(matrix, axes=(0, -1, -2))
