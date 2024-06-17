"""Conditionals."""

from probdiffeq.backend import abc, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util, cond_util


class ConditionalBackend(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, cond1, cond2, /):
        raise NotImplementedError


class ScalarConditional(ConditionalBackend):
    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional

        mean = matrix @ rv.mean + noise.mean
        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky_T = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack)
        return _normal.Normal(mean, cholesky_T.T)

    def revert(self, rv, conditional, /):
        matrix, noise = conditional

        r_ext, (r_bw_p, g_bw_p) = cholesky_util.revert_conditional(
            R_X_F=(matrix @ rv.cholesky).T, R_X=rv.cholesky.T, R_YX=noise.cholesky.T
        )
        m_ext = matrix @ rv.mean + noise.mean
        m_cond = rv.mean - g_bw_p @ m_ext

        marginal = _normal.Normal(m_ext, r_ext.T)
        noise = _normal.Normal(m_cond, r_bw_p.T)
        return marginal, cond_util.Conditional(g_bw_p, noise)

    def apply(self, x, conditional, /):
        matrix, noise = conditional
        matrix = np.squeeze(matrix)
        return _normal.Normal(linalg.vector_dot(matrix, x) + noise.mean, noise.cholesky)

    def merge(self, previous, incoming, /):
        A, b = previous
        C, d = incoming

        g = A @ C
        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        noise = _normal.Normal(xi, Xi)
        return cond_util.Conditional(g, noise)


class DenseConditional(ConditionalBackend):
    def apply(self, x, conditional, /):
        matrix, noise = conditional
        return _normal.Normal(matrix @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matmul, noise = conditional
        R_stack = ((matmul @ rv.cholesky).T, noise.cholesky.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(matmul @ rv.mean + noise.mean, cholesky_new)

    def merge(self, cond1, cond2, /):
        A, b = cond1
        C, d = cond2

        g = A @ C
        xi = A @ d.mean + b.mean
        Xi = cholesky_util.sum_of_sqrtm_factors(
            R_stack=((A @ d.cholesky).T, b.cholesky.T)
        )
        return cond_util.Conditional(g, _normal.Normal(xi, Xi.T))

    def revert(self, rv, conditional, /):
        matrix, noise = conditional
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional(
            R_X_F=(matrix @ cholesky).T, R_X=cholesky.T, R_YX=noise.cholesky.T
        )

        # Gather terms and return
        mean_observed = matrix @ mean + noise.mean
        m_cor = mean - gain @ mean_observed
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(mean_observed, r_obs.T)
        return observed, cond_util.Conditional(gain, corrected)


class IsotropicConditional(ConditionalBackend):
    def apply(self, x, conditional, /):
        A, noise = conditional
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if np.ndim(x) == 1:
            x = x[None, :]
        return _normal.Normal(A @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional

        mean = matrix @ rv.mean + noise.mean

        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(mean, cholesky)

    def merge(self, cond1, cond2, /):
        A, b = cond1
        C, d = cond2

        g = A @ C
        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack).T

        noise = _normal.Normal(xi, Xi)
        return cond_util.Conditional(g, noise)

    def revert(self, rv, conditional, /):
        matrix, noise = conditional

        r_ext_p, (r_bw_p, gain) = cholesky_util.revert_conditional(
            R_X_F=(matrix @ rv.cholesky).T, R_X=rv.cholesky.T, R_YX=noise.cholesky.T
        )
        extrapolated_cholesky = r_ext_p.T
        corrected_cholesky = r_bw_p.T

        extrapolated_mean = matrix @ rv.mean + noise.mean
        corrected_mean = rv.mean - gain @ extrapolated_mean

        extrapolated = _normal.Normal(extrapolated_mean, extrapolated_cholesky)
        corrected = _normal.Normal(corrected_mean, corrected_cholesky)
        return extrapolated, cond_util.Conditional(gain, corrected)
