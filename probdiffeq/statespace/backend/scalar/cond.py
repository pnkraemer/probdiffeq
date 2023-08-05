import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.backend import _cond
from probdiffeq.statespace.backend.scalar import random


class TransformImpl(_cond.ConditionalImpl):
    def apply(self, x, conditional, /):
        raise NotImplementedError

    def marginalise(self, rv, transformation, /):
        # currently, assumes that A(rv.cholesky) is a vector, not a matrix.
        A, b = transformation
        cholesky_new = _sqrt_util.triu_via_qr(A(rv.cholesky)[:, None])
        cholesky_new_squeezed = jnp.reshape(cholesky_new, ())
        return random.Normal(A(rv.mean) + b, cholesky_new_squeezed)

    def revert(self, rv, transformation, /):
        # Assumes that A maps a vector to a scalar...

        # Extract information
        A, b = transformation

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to transformation_revert_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=A(rv.cholesky)[:, None], R_X=rv.cholesky.T
        )
        cov_sqrtm_lower_obs = jnp.reshape(r_obs, ())
        cov_sqrtm_lower_cor = r_cor.T
        gain = jnp.squeeze(gain, axis=-1)

        # Gather terms and return
        m_cor = rv.mean - gain * (A(rv.mean) + b)
        corrected = random.Normal(m_cor, cov_sqrtm_lower_cor)
        observed = random.Normal(A(rv.mean) + b, cov_sqrtm_lower_obs)
        return observed, (corrected, gain)

    def merge(self, cond1, cond2, /):
        raise NotImplementedError


class ConditionalImpl(_cond.ConditionalImpl):
    def marginalise(self, rv, conditional, /):
        A, noise = conditional

        mean = A @ rv.mean
        cholesky_T = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((A @ rv.cholesky).T, noise.cholesky.T)
        )
        return random.Normal(mean, cholesky_T.T)

    def revert(self, rv, conditional, /):
        a, noise = conditional

        r_ext, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
            R_X_F=(a @ rv.cholesky).T,
            R_X=rv.cholesky.T,
            R_YX=(noise.cholesky).T,
        )
        m_ext = a @ rv.mean + noise.mean
        m_cond = rv.mean - g_bw_p @ m_ext

        marginal = random.Normal(m_ext, r_ext.T)
        noise = random.Normal(m_cond, r_bw_p.T)
        return marginal, (g_bw_p, noise)

    def apply(self, x, conditional, /):
        a, noise = conditional
        return random.Normal(a @ x + noise.mean, noise.cholesky)

    def merge(self, previous, incoming, /):
        A, b = previous
        C, d = incoming

        g = A @ C
        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        noise = random.Normal(xi, Xi)
        return g, noise


class ConditionalBackEnd(_cond.ConditionalBackEnd):
    @property
    def conditional(self) -> ConditionalImpl:
        return ConditionalImpl()

    @property
    def transform(self) -> ConditionalImpl:
        return TransformImpl()
