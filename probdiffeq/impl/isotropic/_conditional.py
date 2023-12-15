from probdiffeq.backend import numpy as np
from probdiffeq.impl import _conditional
from probdiffeq.impl.isotropic import _normal
from probdiffeq.util import cholesky_util, cond_util


class ConditionalBackend(_conditional.ConditionalBackend):
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
