from probdiffeq import _sqrt_util
from probdiffeq.backend import _conditional
from probdiffeq.backend.isotropic import _normal


class ConditionalBackend(_conditional.ConditionalBackend):
    def apply(self, x, conditional, /):
        A, noise = conditional
        return _normal.Normal(A @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional

        mean = matrix @ rv.mean + noise.mean

        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(mean, cholesky)

    def merge(self, cond1, cond2, /):
        raise NotImplementedError

    def revert(self, rv, conditional, /):
        raise NotImplementedError
