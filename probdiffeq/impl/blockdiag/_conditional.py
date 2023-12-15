"""Conditional implementation."""
from probdiffeq.backend import functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _conditional
from probdiffeq.impl.blockdiag import _normal
from probdiffeq.util import cholesky_util, cond_util


class ConditionalBackend(_conditional.ConditionalBackend):
    def apply(self, x, conditional, /):
        if np.ndim(x) == 1:
            x = x[..., None]

        def apply_unbatch(m, s, n):
            return _normal.Normal(m @ s + n.mean, n.cholesky)

        matrix, noise = conditional
        return functools.vmap(apply_unbatch)(matrix, x, noise)

    def marginalise(self, rv, conditional, /):
        matrix, noise = conditional
        assert matrix.ndim == 3

        mean = np.einsum("ijk,ik->ij", matrix, rv.mean) + noise.mean

        chol1 = _transpose(matrix @ rv.cholesky)
        chol2 = _transpose(noise.cholesky)
        R_stack = (chol1, chol2)
        cholesky = functools.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack)
        return _normal.Normal(mean, _transpose(cholesky))

    def merge(self, cond1, cond2, /):
        A, b = cond1
        C, d = cond2

        g = A @ C
        xi = (A @ d.mean[..., None])[..., 0] + b.mean
        R_stack = (_transpose(A @ d.cholesky), _transpose(b.cholesky))
        Xi = _transpose(functools.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack))

        noise = _normal.Normal(xi, Xi)
        return cond_util.Conditional(g, noise)

    def revert(self, rv, conditional, /):
        A, noise = conditional
        rv_chol_upper = np.transpose(rv.cholesky, axes=(0, 2, 1))
        noise_chol_upper = np.transpose(noise.cholesky, axes=(0, 2, 1))
        A_rv_chol_upper = np.transpose(A @ rv.cholesky, axes=(0, 2, 1))

        revert = functools.vmap(cholesky_util.revert_conditional)
        r_obs, (r_cor, gain) = revert(A_rv_chol_upper, rv_chol_upper, noise_chol_upper)

        cholesky_obs = np.transpose(r_obs, axes=(0, 2, 1))
        cholesky_cor = np.transpose(r_cor, axes=(0, 2, 1))

        # Gather terms and return
        mean_observed = (A @ rv.mean[..., None])[..., 0] + noise.mean
        m_cor = rv.mean - (gain @ (mean_observed[..., None]))[..., 0]
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))
