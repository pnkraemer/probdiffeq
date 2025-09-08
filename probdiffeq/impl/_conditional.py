"""LatentConds."""

from probdiffeq.backend import abc, containers, functools, linalg, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array
from probdiffeq.impl import _normal, _stats
from probdiffeq.util import cholesky_util


@tree_util.register_dataclass
@containers.dataclass
class LatentCond:
    """LatentCond distributions."""

    A: Array
    noise: _normal.Normal


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
        return LatentCond(matmul, noise)

    @abc.abstractmethod
    def identity(self, ndim, /):
        raise NotImplementedError

    @abc.abstractmethod
    def ibm_transitions(self, num_derivatives, output_scale=None):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, cond, p, p_inv, /):
        raise NotImplementedError

    @abc.abstractmethod
    def to_derivative(self, i, standard_deviation):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_noise(self, cond, scale):
        raise NotImplementedError


class DenseConditional(ConditionalBackend):
    def __init__(self, ode_shape, num_derivatives, unravel, flat_shape):
        self.ode_shape = ode_shape
        self.num_derivatives = num_derivatives
        self.unravel = unravel
        self.flat_shape = flat_shape

    def apply(self, x, cond, /):
        return _normal.Normal(cond.A @ x + cond.noise.mean, cond.noise.cholesky)

    def marginalise(self, rv, cond, /):
        R_stack = ((cond.A @ rv.cholesky).T, cond.noise.cholesky.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(cond.A @ rv.mean + cond.noise.mean, cholesky_new)

    def merge(self, cond1, cond2, /):
        A, b = cond1.A, cond1.noise
        C, d = cond2.A, cond2.noise

        g = A @ C
        xi = A @ d.mean + b.mean
        Xi = cholesky_util.sum_of_sqrtm_factors(
            R_stack=((A @ d.cholesky).T, b.cholesky.T)
        )
        return LatentCond(g, _normal.Normal(xi, Xi.T))

    def revert(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional(
            R_X_F=(matrix @ cholesky).T, R_X=cholesky.T, R_YX=noise.cholesky.T
        )

        # Gather terms and return
        mean_observed = matrix @ mean + noise.mean
        m_cor = mean - gain @ mean_observed
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(mean_observed, r_obs.T)
        return observed, LatentCond(gain, corrected)

    def identity(self, ndim, /) -> LatentCond:
        (d,) = self.ode_shape
        n = ndim * d

        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        return LatentCond(A, _normal.Normal(m, C))

    def ibm_transitions(self, base_scale):
        a, q_sqrtm = system_matrices_1d(self.num_derivatives)
        (d,) = self.ode_shape

        eye_d = np.eye(d)
        A = np.kron(a, eye_d)
        Q = np.kron(q_sqrtm, eye_d)

        q0 = np.zeros(self.flat_shape)

        precon_fun = preconditioner_prepare(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            scale = base_scale * output_scale
            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            noise = _normal.Normal(q0, scale * Q)
            return LatentCond(A, noise), (p, p_inv)

        return discretise

    def preconditioner_apply(self, cond, p, p_inv, /):
        normal = _normal.DenseNormal(ode_shape=self.ode_shape)
        noise = normal.preconditioner_apply(cond.noise, p)
        A = p[:, None] * cond.A * p_inv[None, :]
        return LatentCond(A, noise)

    def to_derivative(self, i, standard_deviation):
        x = np.zeros(self.flat_shape)

        def select(a):
            return self.unravel(a)[i]

        linop = functools.jacrev(select)(x)

        (d,) = self.ode_shape
        bias = np.zeros((d,))
        eye = np.eye(d)
        noise = _normal.Normal(bias, standard_deviation * eye)
        return LatentCond(linop, noise)

    def rescale_noise(self, cond, scale):
        A = cond.A
        noise = cond.noise
        stats = _stats.DenseStats(ode_shape=self.ode_shape, unravel=self.unravel)
        noise_new = stats.rescale_cholesky(noise, scale)
        return LatentCond(A, noise_new)


class IsotropicConditional(ConditionalBackend):
    def __init__(self, *, ode_shape, num_derivatives, unravel_tree):
        self.ode_shape = ode_shape
        self.num_derivatives = num_derivatives
        self.unravel_tree = unravel_tree

    def apply(self, x, cond, /):
        A, noise = cond.A, cond.noise
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if np.ndim(x) == 1:
            x = x[None, :]
        return _normal.Normal(A @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise

        mean = matrix @ rv.mean + noise.mean

        R_stack = ((matrix @ rv.cholesky).T, noise.cholesky.T)
        cholesky = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return _normal.Normal(mean, cholesky)

    def merge(self, cond1, cond2, /):
        A, b = cond1.A, cond1.noise
        C, d = cond2.A, cond2.noise

        g = A @ C
        xi = A @ d.mean + b.mean
        R_stack = ((A @ d.cholesky).T, b.cholesky.T)
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack).T

        noise = _normal.Normal(xi, Xi)
        return LatentCond(g, noise)

    def revert(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise

        r_ext_p, (r_bw_p, gain) = cholesky_util.revert_conditional(
            R_X_F=(matrix @ rv.cholesky).T, R_X=rv.cholesky.T, R_YX=noise.cholesky.T
        )
        extrapolated_cholesky = r_ext_p.T
        corrected_cholesky = r_bw_p.T

        extrapolated_mean = matrix @ rv.mean + noise.mean
        corrected_mean = rv.mean - gain @ extrapolated_mean

        extrapolated = _normal.Normal(extrapolated_mean, extrapolated_cholesky)
        corrected = _normal.Normal(corrected_mean, corrected_cholesky)
        return extrapolated, LatentCond(gain, corrected)

    def identity(self, num_hidden_states_per_ode_dim, /) -> LatentCond:
        m0 = np.zeros((num_hidden_states_per_ode_dim, *self.ode_shape))
        c0 = np.zeros((num_hidden_states_per_ode_dim, num_hidden_states_per_ode_dim))
        noise = _normal.Normal(m0, c0)
        matrix = np.eye(num_hidden_states_per_ode_dim)
        return LatentCond(matrix, noise)

    def ibm_transitions(self, base_scale):
        A, q_sqrtm = system_matrices_1d(self.num_derivatives)
        q0 = np.zeros((self.num_derivatives + 1, *self.ode_shape))
        precon_fun = preconditioner_prepare(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            scale = base_scale * output_scale
            p, p_inv = precon_fun(dt)
            noise = _normal.Normal(q0, scale * q_sqrtm)
            return LatentCond(A, noise), (p, p_inv)

        return discretise

    def preconditioner_apply(self, cond, p, p_inv, /):
        A, noise = cond.A, cond.noise

        A_new = p[:, None] * A * p_inv[None, :]

        noise = _normal.Normal(p[:, None] * noise.mean, p[:, None] * noise.cholesky)
        return LatentCond(A_new, noise)

    def to_derivative(self, i, standard_deviation):
        def select(a):
            return tree_util.ravel_pytree(self.unravel_tree(a)[i])[0]

        m = np.zeros((self.num_derivatives + 1,))
        linop = functools.jacrev(select)(m)

        bias = np.zeros(self.ode_shape)
        eye = np.eye(1)
        noise = _normal.Normal(bias, standard_deviation * eye)

        return LatentCond(linop, noise)

    def rescale_noise(self, cond, scale):
        A = cond.A
        noise = cond.noise
        stats = _stats.IsotropicStats(
            ode_shape=self.ode_shape, unravel=self.unravel_tree
        )
        noise_new = stats.rescale_cholesky(noise, scale)
        return LatentCond(A, noise_new)


class BlockDiagConditional(ConditionalBackend):
    def __init__(self, *, ode_shape, num_derivatives, unravel_tree):
        self.ode_shape = ode_shape
        self.num_derivatives = num_derivatives
        self.unravel_tree = unravel_tree

    def apply(self, x, cond, /):
        if np.ndim(x) == 1:
            x = x[..., None]

        def apply_unbatch(m, s, n):
            return _normal.Normal(m @ s + n.mean, n.cholesky)

        matrix, noise = cond.A, cond.noise
        return functools.vmap(apply_unbatch)(matrix, x, noise)

    def marginalise(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise
        assert matrix.ndim == 3

        mean = np.einsum("ijk,ik->ij", matrix, rv.mean) + noise.mean

        chol1 = _transpose(matrix @ rv.cholesky)
        chol2 = _transpose(noise.cholesky)
        R_stack = (chol1, chol2)
        cholesky = functools.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack)
        return _normal.Normal(mean, _transpose(cholesky))

    def merge(self, cond1, cond2, /):
        A, b = cond1.A, cond1.noise
        C, d = cond2.A, cond2.noise

        g = A @ C
        xi = (A @ d.mean[..., None])[..., 0] + b.mean
        R_stack = (_transpose(A @ d.cholesky), _transpose(b.cholesky))
        Xi = _transpose(functools.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack))

        noise = _normal.Normal(xi, Xi)
        return LatentCond(g, noise)

    def revert(self, rv, cond, /):
        A, noise = cond.A, cond.noise
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
        return observed, LatentCond(gain, corrected)

    def identity(self, ndim, /) -> LatentCond:
        m0 = np.zeros((*self.ode_shape, ndim))
        c0 = np.zeros((*self.ode_shape, ndim, ndim))
        noise = _normal.Normal(m0, c0)

        matrix = np.ones((*self.ode_shape, 1, 1)) * np.eye(ndim, ndim)[None, ...]
        return LatentCond(matrix, noise)

    def ibm_transitions(self, base_scale):
        a, q_sqrtm = system_matrices_1d(self.num_derivatives)
        q0 = np.zeros((self.num_derivatives + 1,))
        precon_fun = preconditioner_prepare(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            p, p_inv = precon_fun(dt)
            scale = base_scale * output_scale
            cond = functools.vmap(discretise_1d)(scale)
            return cond, (p, p_inv)

        def discretise_1d(scale):
            noise = _normal.Normal(q0, scale * q_sqrtm)
            return LatentCond(a, noise)

        return discretise

    def preconditioner_apply(self, cond, p, p_inv, /):
        A_new = p[None, :, None] * cond.A * p_inv[None, None, :]

        normal = _normal.BlockDiagNormal(ode_shape=self.ode_shape)
        noise = normal.preconditioner_apply(cond.noise, p)
        return LatentCond(A_new, noise)

    def to_derivative(self, i, standard_deviation):
        def select(a):
            return tree_util.ravel_pytree(self.unravel_tree(a)[i])[0]

        x = np.zeros((*self.ode_shape, self.num_derivatives + 1))
        linop = functools.vmap(functools.jacrev(select))(x)

        bias = np.zeros((*self.ode_shape, 1))
        eye = np.ones((*self.ode_shape, 1, 1)) * np.eye(1)[None, ...]
        noise = _normal.Normal(bias, standard_deviation * eye)

        return LatentCond(linop, noise)

    def rescale_noise(self, cond, scale):
        A = cond.A
        noise = cond.noise
        stats = _stats.BlockDiagStats(
            ode_shape=self.ode_shape, unravel=self.unravel_tree
        )
        noise_new = stats.rescale_cholesky(noise, scale)
        return LatentCond(A, noise_new)


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))


def system_matrices_1d(num_derivatives):
    """Construct the IBM system matrices."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...

    # Cholesky factor of flip(hilbert(n))
    Q_1d = cholesky_util.cholesky_hilbert(num_derivatives + 1)
    Q_1d_flipped = np.flip(Q_1d, axis=0)
    Q_1d = linalg.qr_r(Q_1d_flipped.T).T
    return A_1d, Q_1d


def preconditioner_diagonal(dt, *, scales, powers):
    """Construct the diagonal IBM preconditioner."""
    dt_abs = np.abs(dt)
    scaling_vector = np.power(dt_abs, powers) / scales
    scaling_vector_inv = np.power(dt_abs, -powers) * scales
    return scaling_vector, scaling_vector_inv


def preconditioner_prepare(*, num_derivatives):
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)
    powers = powers + 0.5
    return tree_util.Partial(preconditioner_diagonal, scales=scales, powers=powers)


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = functools.vmap(k, in_axes=(0, None), out_axes=-1)
    return functools.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))
