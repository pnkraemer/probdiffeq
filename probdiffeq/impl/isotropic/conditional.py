from probdiffeq import _sqrt_util
from probdiffeq.impl import _conditional, matfree
from probdiffeq.impl.isotropic import _normal


class ConditionalBackend(_conditional.ConditionalBackend):
    def apply(self, x, conditional, /):
        A, noise = conditional
        assert isinstance(A, matfree.LinOp)

        return _normal.Normal(A @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional
        assert isinstance(matrix, matfree.LinOp)

        mean = matrix @ rv.mean + noise.mean

        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(mean, cholesky)

    def merge(self, cond1, cond2, /):
        A, b = cond1
        C, d = cond2

        g = matfree.merge_linops(A, C)

        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack).T

        noise = _normal.Normal(xi, Xi)
        return g, noise

    def revert(self, rv, conditional, /):
        matrix, noise = conditional
        assert isinstance(matrix, matfree.LinOp)

        r_ext_p, (r_bw_p, gain) = _sqrt_util.revert_conditional(
            R_X_F=(matrix @ rv.cholesky).T,
            R_X=rv.cholesky.T,
            R_YX=noise.cholesky.T,
        )
        extrapolated_cholesky = r_ext_p.T
        corrected_cholesky = r_bw_p.T

        extrapolated_mean = matrix @ rv.mean + noise.mean
        corrected_mean = rv.mean - gain @ extrapolated_mean

        extrapolated = _normal.Normal(extrapolated_mean, extrapolated_cholesky)
        corrected = _normal.Normal(corrected_mean, corrected_cholesky)
        return extrapolated, (matfree.linop_from_matmul(gain), corrected)
