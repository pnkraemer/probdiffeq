"""Conditionals."""

from probdiffeq.backend import abc, containers, functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util


class Conditional(containers.NamedTuple):
    """Conditional distributions."""

    matmul: Array  # or anything with a __matmul__ implementation
    noise: Any  # Usually a random-variable type


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

    def conditional(self, matmul, noise):
        return Conditional(matmul, noise)

    @abc.abstractmethod
    def identity(self, num_derivatives_per_ode_dimension, /):
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
        return marginal, Conditional(g_bw_p, noise)

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
        return Conditional(g, noise)

    def identity(self, ndim, /) -> Conditional:
        transition = np.eye(ndim)
        mean = np.zeros((ndim,))
        cov_sqrtm = np.zeros((ndim, ndim))
        noise = _normal.Normal(mean, cov_sqrtm)
        return Conditional(transition, noise)


class DenseConditional(ConditionalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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
        return Conditional(g, _normal.Normal(xi, Xi.T))

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
        return observed, Conditional(gain, corrected)

    def identity(self, ndim, /) -> Conditional:
        (d,) = self.ode_shape
        n = ndim * d

        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        return Conditional(A, _normal.Normal(m, C))


class IsotropicConditional(ConditionalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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
        return Conditional(g, noise)

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
        return extrapolated, Conditional(gain, corrected)

    def identity(self, num_hidden_states_per_ode_dim, /) -> Conditional:
        m0 = np.zeros((num_hidden_states_per_ode_dim, *self.ode_shape))
        c0 = np.zeros((num_hidden_states_per_ode_dim, num_hidden_states_per_ode_dim))
        noise = _normal.Normal(m0, c0)
        matrix = np.eye(num_hidden_states_per_ode_dim)
        return Conditional(matrix, noise)


class BlockDiagConditional(ConditionalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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
        return Conditional(g, noise)

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
        return observed, Conditional(gain, corrected)

    def identity(self, ndim, /) -> Conditional:
        m0 = np.zeros((*self.ode_shape, ndim))
        c0 = np.zeros((*self.ode_shape, ndim, ndim))
        noise = _normal.Normal(m0, c0)

        matrix = np.ones((*self.ode_shape, 1, 1)) * np.eye(ndim, ndim)[None, ...]
        return Conditional(matrix, noise)


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))
