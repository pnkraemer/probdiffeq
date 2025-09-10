"""LatentConds."""

from probdiffeq.backend import abc, containers, functools, linalg, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array
from probdiffeq.impl import _normal, _stats
from probdiffeq.util import cholesky_util


@tree_util.register_dataclass
@containers.dataclass
class LatentCond:
    """Conditional distributions in latent space."""

    A: Array
    """Linear operator in latent space."""

    noise: _normal.Normal
    """Additive Gaussian noise in latent space."""

    to_latent: Array
    """Pull an observed vector into the latent space."""

    to_observed: Array
    """Push a latent vector into the observed space."""


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

    @abc.abstractmethod
    def identity(self, ndim, /):
        raise NotImplementedError

    @abc.abstractmethod
    def ibm_transitions(self, num_derivatives, output_scale=None):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, cond, /):
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
        x = cond.to_latent * x
        mean = cond.to_observed * (cond.A @ x + cond.noise.mean)
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky
        return _normal.Normal(mean, cholesky)

    def marginalise(self, rv, cond, /):
        mean = cond.to_latent * rv.mean
        cholesky = cond.to_latent[:, None] * rv.cholesky

        R_stack = ((cond.A @ cholesky).T, cond.noise.cholesky.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = cond.to_observed * (cond.A @ mean + cond.noise.mean)
        cholesky_new = cond.to_observed[:, None] * cholesky_new
        return _normal.Normal(mean_new, cholesky_new)

    def merge(self, cond1: LatentCond, cond2: LatentCond, /) -> LatentCond:
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        g = cond1.A @ (T[:, None] * cond2.A)

        # Combined mean
        xi = cond1.A @ (T * cond2.noise.mean) + cond1.noise.mean

        # Cholesky factor of combined covariance
        R1 = (cond1.A @ (T[:, None] * cond2.noise.cholesky)).T
        R2 = cond1.noise.cholesky.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        # Gather and return
        noise = _normal.Normal(xi, Xi.T)
        return LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv: _normal.Normal, cond: LatentCond, /):
        # Pull RV into the latent space
        mean = cond.to_latent * rv.mean
        cholesky = cond.to_latent[:, None] * rv.cholesky

        # QR-decomposition
        R_X_F, R_X, R_YX = (cond.A @ cholesky).T, cholesky.T, cond.noise.cholesky.T
        tmp = cholesky_util.revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)
        r_obs, (r_cor, gain) = tmp

        # Push correction into observed space
        mean_observed = cond.A @ mean + cond.noise.mean
        mean_corrected = mean - gain @ mean_observed
        cholesky_corrected = r_cor.T
        corrected = _normal.Normal(mean_corrected, cholesky_corrected)
        cond_new = LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather the observed variable
        mean = cond.to_observed * mean_observed
        cholesky = cond.to_observed[:, None] * r_obs.T
        observed = _normal.Normal(mean, cholesky)
        return observed, cond_new

    def identity(self, ndim, /) -> LatentCond:
        (d,) = self.ode_shape
        n = ndim * d

        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        noise = _normal.Normal(m, C)
        ones = np.ones((n,))
        return LatentCond(A, noise, to_latent=ones, to_observed=ones)

    def ibm_transitions(self, base_scale):
        a, q_sqrtm = wiener_integrated_system_matrices_1d(self.num_derivatives)
        (d,) = self.ode_shape

        eye_d = np.eye(d)
        A = np.kron(a, eye_d)
        Q = np.kron(q_sqrtm, eye_d)

        q0 = np.zeros(self.flat_shape)

        precon_fun = preconditioner_taylor(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            scale = base_scale * output_scale
            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            noise = _normal.Normal(q0, scale * Q)
            return LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[:, None] * cond.A * cond.to_latent[None, :]
        mean = cond.to_observed * cond.noise.mean
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky
        noise = _normal.Normal(mean, cholesky)

        to_observed = np.ones_like(cond.to_observed)
        to_latent = np.ones_like(cond.to_latent)
        return LatentCond(A, noise, to_observed=to_observed, to_latent=to_latent)

    def to_derivative(self, i, u, standard_deviation):
        def select(a):
            return tree_util.ravel_pytree(self.unravel(a)[i])[0]

        x = np.zeros(self.flat_shape)
        linop = functools.jacrev(select)(x)

        u_flat, _ = tree_util.ravel_pytree(u)
        stdev, _ = tree_util.ravel_pytree(standard_deviation)
        assert stdev.shape == (1,)

        (s,) = stdev
        cholesky = np.eye(len(u_flat)) * s
        noise = _normal.Normal(-u_flat, cholesky)

        to_latent = np.ones(linop.shape[1])
        to_observed = np.ones(linop.shape[0])
        return LatentCond(linop, noise, to_latent=to_latent, to_observed=to_observed)

    def rescale_noise(self, cond, scale):
        A = cond.A
        noise = cond.noise
        stats = _stats.DenseStats(ode_shape=self.ode_shape, unravel=self.unravel)
        noise_new = stats.rescale_cholesky(noise, scale)
        return LatentCond(
            A, noise_new, to_latent=cond.to_latent, to_observed=cond.to_observed
        )


class IsotropicConditional(ConditionalBackend):
    def __init__(self, *, ode_shape, num_derivatives, unravel_tree):
        self.ode_shape = ode_shape
        self.num_derivatives = num_derivatives
        self.unravel_tree = unravel_tree

    def apply(self, x, cond, /):
        # TODO: is this still relevant?
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if np.ndim(x) == 1:
            x = x[None, :]

        x = cond.to_latent[:, None] * x
        mean_new = cond.to_observed[:, None] * cond.A @ x + cond.noise.mean
        cholesky_new = cond.to_observed[:, None] * cond.noise.cholesky
        return _normal.Normal(mean_new, cholesky_new)

    def marginalise(self, rv, cond, /):
        mean = cond.to_latent[:, None] * rv.mean
        cholesky = cond.to_latent[:, None] * rv.cholesky
        R_stack = ((cond.A @ cholesky).T, cond.noise.cholesky.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = cond.to_observed[:, None] * (cond.A @ mean + cond.noise.mean)
        cholesky_new = cond.to_observed[:, None] * cholesky_new
        return _normal.Normal(mean_new, cholesky_new)

    def merge(self, cond1, cond2, /):
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        g = cond1.A @ (T[:, None] * cond2.A)

        # Combined mean
        xi = cond1.A @ (T[:, None] * cond2.noise.mean) + cond1.noise.mean

        # Cholesky factor of combined covariance
        R1 = (cond1.A @ (T[:, None] * cond2.noise.cholesky)).T
        R2 = cond1.noise.cholesky.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        # Gather and return
        noise = _normal.Normal(xi, Xi.T)
        return LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv, cond, /):
        # Pull RV into the latent space
        mean = cond.to_latent[:, None] * rv.mean
        cholesky = cond.to_latent[:, None] * rv.cholesky

        # QR-decomposition
        R_X_F, R_X, R_YX = (cond.A @ cholesky).T, cholesky.T, cond.noise.cholesky.T
        tmp = cholesky_util.revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)
        r_obs, (r_cor, gain) = tmp

        # Push correction into observed space
        mean_observed = cond.A @ mean + cond.noise.mean
        mean_corrected = mean - gain @ mean_observed
        cholesky_corrected = r_cor.T
        corrected = _normal.Normal(mean_corrected, cholesky_corrected)
        cond_new = LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather the observed variable
        mean = cond.to_observed[:, None] * mean_observed
        cholesky = cond.to_observed[:, None] * r_obs.T
        observed = _normal.Normal(mean, cholesky)
        return observed, cond_new

    def identity(self, num, /) -> LatentCond:
        m0 = np.zeros((num, *self.ode_shape))
        c0 = np.zeros((num, num))
        noise = _normal.Normal(m0, c0)
        matrix = np.eye(num)
        ones = np.ones((num,))
        return LatentCond(matrix, noise, to_latent=ones, to_observed=ones)

    def ibm_transitions(self, base_scale):
        A, q_sqrtm = wiener_integrated_system_matrices_1d(self.num_derivatives)
        q0 = np.zeros((self.num_derivatives + 1, *self.ode_shape))
        precon_fun = preconditioner_taylor(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            scale = base_scale * output_scale
            p, p_inv = precon_fun(dt)
            noise = _normal.Normal(q0, scale * q_sqrtm)
            return LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[:, None] * cond.A * cond.to_latent[None, :]
        mean = cond.to_observed[:, None] * cond.noise.mean
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky
        noise = _normal.Normal(mean, cholesky)
        to_observed = np.ones_like(cond.to_observed)
        to_latent = np.ones_like(cond.to_latent)
        return LatentCond(A, noise, to_observed=to_observed, to_latent=to_latent)

    def to_derivative(self, i, u, standard_deviation):
        def select(a):
            return tree_util.ravel_pytree(self.unravel_tree(a)[i])[0]

        m = np.zeros((self.num_derivatives + 1,))
        linop = functools.jacrev(select)(m)

        u_flat, _ = tree_util.ravel_pytree(u)

        stdev, _ = tree_util.ravel_pytree(standard_deviation)

        assert stdev.shape == (1,)
        cholesky = linalg.diagonal_matrix(stdev)
        noise = _normal.Normal(-u_flat, cholesky)

        to_latent = np.ones(linop.shape[1])
        to_observed = np.ones(linop.shape[0])
        return LatentCond(linop, noise, to_latent=to_latent, to_observed=to_observed)

    def rescale_noise(self, cond, scale):
        stats = _stats.IsotropicStats(
            ode_shape=self.ode_shape, unravel=self.unravel_tree
        )
        noise_new = stats.rescale_cholesky(cond.noise, scale)
        return LatentCond(
            cond.A, noise_new, to_latent=cond.to_latent, to_observed=cond.to_observed
        )


class BlockDiagConditional(ConditionalBackend):
    def __init__(self, *, ode_shape, num_derivatives, unravel_tree):
        self.ode_shape = ode_shape
        self.num_derivatives = num_derivatives
        self.unravel_tree = unravel_tree

    def apply(self, x, cond, /):
        if np.ndim(x) == 1:
            x = x[..., None]

        def apply_unbatch(m, s, n):
            s = cond.to_latent * s
            m_new = cond.to_observed * (m @ s + n.mean)
            c_new = cond.to_observed[:, None] * n.cholesky
            return _normal.Normal(m_new, c_new)

        matrix, noise = cond.A, cond.noise
        return functools.vmap(apply_unbatch)(matrix, x, noise)

    def marginalise(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise
        assert matrix.ndim == 3
        mean = cond.to_latent[None, :] * rv.mean
        cholesky = cond.to_latent[None, :, None] * rv.cholesky

        mean_marg = np.einsum("ijk,ik->ij", matrix, mean) + noise.mean

        chol1 = _transpose(matrix @ cholesky)
        chol2 = _transpose(noise.cholesky)
        R_stack = (chol1, chol2)
        cholesky = functools.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack)

        mean_new = cond.to_observed[None, :] * mean_marg
        cholesky_new = cond.to_observed[None, :, None] * _transpose(cholesky)
        return _normal.Normal(mean_new, cholesky_new)

    def merge(self, cond1, cond2, /):
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        A1, A2 = cond1.A, T[None, :, None] * cond2.A
        g = functools.vmap(lambda a, b: a @ b)(A1, A2)

        # Combined mean
        m1, m2 = T[None, :] * cond2.noise.mean, cond1.noise.mean
        xi = functools.vmap(lambda a, b, c: a @ b + c)(A1, m1, m2)

        # Cholesky factor of combined covariance
        C1, C2 = cond1.noise.cholesky, T[None, :, None] * cond2.noise.cholesky
        R1 = _transpose(functools.vmap(lambda a, b: a @ b)(A1, C2))
        R2 = _transpose(C1)
        Xi = functools.vmap(cholesky_util.sum_of_sqrtm_factors)((R1, R2))
        Xi = _transpose(Xi)

        # Gather and return
        noise = _normal.Normal(xi, Xi)
        return LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv, cond, /):
        # Pull RV into latent space
        mean = cond.to_latent[None, :] * rv.mean
        cholesky = cond.to_latent[None, :, None] * rv.cholesky

        # QR decomposition
        rv_chol_upper = _transpose(cholesky)
        noise_chol_upper = _transpose(cond.noise.cholesky)
        A_rv_chol_upper = _transpose(cond.A @ cholesky)
        revert = functools.vmap(cholesky_util.revert_conditional)
        r_obs, (r_cor, gain) = revert(A_rv_chol_upper, rv_chol_upper, noise_chol_upper)
        cholesky_obs = np.transpose(r_obs, axes=(0, 2, 1))
        cholesky_cor = np.transpose(r_cor, axes=(0, 2, 1))

        # New backward conditional
        mean_observed = (cond.A @ mean[..., None])[..., 0] + cond.noise.mean
        mean_corrected = mean - (gain @ (mean_observed[..., None]))[..., 0]
        corrected = _normal.Normal(mean_corrected, cholesky_cor)
        bwd = LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather observed RV
        mean_observed = cond.to_observed[None, :] * mean_observed
        cholesky_observed = cond.to_observed[None, :, None] * cholesky_obs
        observed = _normal.Normal(mean_observed, cholesky_observed)
        return observed, bwd

    def identity(self, ndim, /) -> LatentCond:
        m0 = np.zeros((*self.ode_shape, ndim))
        c0 = np.zeros((*self.ode_shape, ndim, ndim))
        noise = _normal.Normal(m0, c0)
        matrix = np.ones((*self.ode_shape, 1, 1)) * np.eye(ndim, ndim)[None, ...]
        ones = np.ones((ndim,))
        return LatentCond(matrix, noise, to_latent=ones, to_observed=ones)

    def ibm_transitions(self, base_scale):
        a, q_sqrtm = wiener_integrated_system_matrices_1d(self.num_derivatives)
        q0 = np.zeros((self.num_derivatives + 1,))
        precon_fun = preconditioner_taylor(num_derivatives=self.num_derivatives)

        def discretise(dt, output_scale):
            p, p_inv = precon_fun(dt)
            scale = base_scale * output_scale
            A_batch, noise_batch = functools.vmap(discretise_1d)(scale)
            return LatentCond(A_batch, noise_batch, to_latent=p_inv, to_observed=p)

        def discretise_1d(scale):
            noise = _normal.Normal(q0, scale * q_sqrtm)
            return a, noise

        return discretise

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[None, :, None] * cond.A * cond.to_latent[None, None, :]
        mean = cond.to_observed[None, :] * cond.noise.mean
        cholesky = cond.to_observed[None, :, None] * cond.noise.cholesky
        noise = _normal.Normal(mean, cholesky)
        to_observed = np.ones_like(cond.to_observed)
        to_latent = np.ones_like(cond.to_latent)
        return LatentCond(A, noise, to_observed=to_observed, to_latent=to_latent)

    def to_derivative(self, i, u, standard_deviation):
        def select(a):
            return tree_util.ravel_pytree(self.unravel_tree(a)[i])[0]

        x = np.zeros((*self.ode_shape, self.num_derivatives + 1))
        linop = functools.vmap(functools.jacrev(select))(x)

        u_flat, _ = tree_util.ravel_pytree(u)
        bias = u_flat[:, None]

        eye = np.ones((*self.ode_shape, 1, 1)) * np.eye(1)[None, ...]
        stdev, _ = tree_util.ravel_pytree(standard_deviation)
        assert stdev.shape == (1,)
        (s,) = stdev
        cholesky = eye * s
        noise = _normal.Normal(-bias, cholesky)
        to_latent = np.ones((linop.shape[2],))
        to_observed = np.ones((linop.shape[1],))
        return LatentCond(linop, noise, to_latent=to_latent, to_observed=to_observed)

    def rescale_noise(self, cond, scale):
        stats = _stats.BlockDiagStats(
            ode_shape=self.ode_shape, unravel=self.unravel_tree
        )
        noise_new = stats.rescale_cholesky(cond.noise, scale)
        return LatentCond(
            cond.A, noise_new, to_latent=cond.to_latent, to_observed=cond.to_observed
        )


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))


def wiener_integrated_system_matrices_1d(num_derivatives):
    """Construct the IBM system matrices."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...

    # Cholesky factor of flip(hilbert(n))
    Q_1d = cholesky_util.cholesky_hilbert(num_derivatives + 1)
    Q_1d_flipped = np.flip(Q_1d, axis=0)
    Q_1d = linalg.qr_r(Q_1d_flipped.T).T
    return A_1d, Q_1d


def preconditioner_taylor(*, num_derivatives):
    """Construct the diagonal preconditioner for Taylor-coefficient state spaces."""
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)
    powers = powers + 0.5

    def precon(dt):
        dt_abs = np.abs(dt)
        scaling_vector = np.power(dt_abs, powers) / scales
        scaling_vector_inv = np.power(dt_abs, -powers) * scales
        return scaling_vector, scaling_vector_inv

    return precon


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = functools.vmap(k, in_axes=(0, None), out_axes=-1)
    return functools.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))
