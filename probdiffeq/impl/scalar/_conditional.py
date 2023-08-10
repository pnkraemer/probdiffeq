"""Conditionals."""
from probdiffeq.impl import _cond_util, _conditional, _sqrt_util
from probdiffeq.impl.scalar import _normal


class ConditionalBackend(_conditional.ConditionalBackend):
    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional

        mean = matrix @ rv.mean + noise.mean
        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky_T = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack)
        return _normal.Normal(mean, cholesky_T.T)

    def revert(self, rv, conditional, /):
        matrix, noise = conditional

        r_ext, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
            R_X_F=(matrix @ rv.cholesky).T,
            R_X=rv.cholesky.T,
            R_YX=noise.cholesky.T,
        )
        m_ext = matrix @ rv.mean + noise.mean
        m_cond = rv.mean - g_bw_p @ m_ext

        marginal = _normal.Normal(m_ext, r_ext.T)
        noise = _normal.Normal(m_cond, r_bw_p.T)
        return marginal, _cond_util.Conditional(g_bw_p, noise)

    def apply(self, x, conditional, /):
        matrix, noise = conditional
        return _normal.Normal(matrix @ x + noise.mean, noise.cholesky)

    def merge(self, previous, incoming, /):
        A, b = previous
        C, d = incoming

        g = A @ C
        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        noise = _normal.Normal(xi, Xi)
        return _cond_util.Conditional(g, noise)
