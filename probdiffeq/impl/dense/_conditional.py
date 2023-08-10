"""Conditional implementation."""
from probdiffeq.impl import _cond_util, _conditional, _sqrt_util
from probdiffeq.impl.dense import _normal


class ConditionalBackend(_conditional.ConditionalBackend):
    def apply(self, x, conditional, /):
        matrix, noise = conditional
        return _normal.Normal(matrix @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matmul, noise = conditional
        R_stack = ((matmul @ rv.cholesky).T, noise.cholesky.T)
        cholesky_new = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(matmul @ rv.mean + noise.mean, cholesky_new)

    def merge(self, cond1, cond2, /):
        A, b = cond1
        C, d = cond2

        g = A @ C
        xi = A @ d.mean + b.mean
        Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=((A @ d.cholesky).T, b.cholesky.T))
        return _cond_util.Conditional(g, _normal.Normal(xi, Xi.T))

    def revert(self, rv, conditional, /):
        matrix, noise = conditional
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
            R_X_F=(matrix @ cholesky).T, R_X=cholesky.T, R_YX=noise.cholesky.T
        )

        # Gather terms and return
        mean_observed = matrix @ mean + noise.mean
        m_cor = mean - gain @ mean_observed
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(mean_observed, r_obs.T)
        return observed, _cond_util.Conditional(gain, corrected)
