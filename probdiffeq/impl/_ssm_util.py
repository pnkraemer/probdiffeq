"""SSM utilities."""

from probdiffeq.backend import abc, functools, linalg, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _conditional, _normal
from probdiffeq.util import cholesky_util


class SSMUtilBackend(abc.ABC):
    @abc.abstractmethod
    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, rv, p, /):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        raise NotImplementedError

    @abc.abstractmethod
    def ibm_transitions(self, num_derivatives, output_scale=None):
        raise NotImplementedError

    # TODO: rename to avoid confusion with conditionals?
    @abc.abstractmethod
    def update_mean(self, mean, x, /, num):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_normal(self, num_derivatives_per_ode_dimension, /, output_scale):
        raise NotImplementedError


class ScalarSSMUtil(SSMUtilBackend):
    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = np.stack(tcoeffs)
        m0_corrected = np.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = np.zeros((num_derivatives + 1, num_derivatives + 1))
        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return _normal.Normal(p * rv.mean, p[:, None] * rv.cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        A = p[:, None] * A * p_inv[None, :]
        noise = _normal.Normal(p * noise.mean, p[:, None] * noise.cholesky)
        return _conditional.Conditional(A, noise)

    def ibm_transitions(self, num_derivatives, output_scale=1.0):
        a, q_sqrtm = system_matrices_1d(num_derivatives, output_scale)
        q0 = np.zeros((num_derivatives + 1,))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return _conditional.Conditional(a, noise), (p, p_inv)

        return discretise

    def standard_normal(self, ndim, /, output_scale):
        mean = np.zeros((ndim,))
        cholesky = output_scale * np.eye(ndim)
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)


class DenseSSMUtil(SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives, output_scale):
        a, q_sqrtm = system_matrices_1d(num_derivatives, output_scale)
        (d,) = self.ode_shape
        eye_d = np.eye(d)
        A = np.kron(eye_d, a)
        Q = np.kron(eye_d, q_sqrtm)

        ndim = d * (num_derivatives + 1)
        q0 = np.zeros((ndim,))
        noise = _normal.Normal(q0, Q)

        precon_fun = preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            p = np.tile(p, d)
            p_inv = np.tile(p_inv, d)
            return _conditional.Conditional(A, noise), (p, p_inv)

        return discretise

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = f"The number of Taylor coefficients {len(tcoeffs)} does not match "
            msg2 = f"the number of derivatives {num_derivatives+1} in the solver."
            raise ValueError(msg1 + msg2)

        if tcoeffs[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = np.stack(tcoeffs)
        m0_corrected = np.reshape(m0_matrix, (-1,), order="F")

        (ode_dim,) = self.ode_shape
        ndim = (num_derivatives + 1) * ode_dim
        c_sqrtm0_corrected = np.zeros((ndim, ndim))

        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        mean = p * rv.mean
        cholesky = p[:, None] * rv.cholesky
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        noise = self.preconditioner_apply(noise, p)
        A = p[:, None] * A * p_inv[None, :]
        return _conditional.Conditional(A, noise)

    def standard_normal(self, ndim, /, output_scale):
        eye_n = np.eye(ndim)
        eye_d = output_scale * np.eye(*self.ode_shape)
        cholesky = np.kron(eye_d, eye_n)
        mean = np.zeros((*self.ode_shape, ndim)).reshape((-1,), order="F")
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        return cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x) / np.sqrt(
            num + 1
        )


class IsotropicSSMUtil(SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives, output_scale):
        A, q_sqrtm = system_matrices_1d(num_derivatives, output_scale)
        q0 = np.zeros((num_derivatives + 1, *self.ode_shape))
        noise = _normal.Normal(q0, q_sqrtm)
        precon_fun = preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return _conditional.Conditional(A, noise), (p, p_inv)

        return discretise

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = f"The number of Taylor coefficients {len(tcoeffs)} does not match "
            msg2 = f"the number of derivatives {num_derivatives+1} in the solver."
            raise ValueError(msg1 + msg2)

        c_sqrtm0_corrected = np.zeros((num_derivatives + 1, num_derivatives + 1))
        m0_corrected = np.stack(tcoeffs)
        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return _normal.Normal(p[:, None] * rv.mean, p[:, None] * rv.cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond

        A_new = p[:, None] * A * p_inv[None, :]

        noise = _normal.Normal(p[:, None] * noise.mean, p[:, None] * noise.cholesky)
        return _conditional.Conditional(A_new, noise)

    def standard_normal(self, num, /, output_scale):
        mean = np.zeros((num, *self.ode_shape))
        cholesky = output_scale * np.eye(num)
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)


class BlockDiagSSMUtil(SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives, output_scale):
        system_matrices = functools.vmap(system_matrices_1d, in_axes=(None, 0))
        a, q_sqrtm = system_matrices(num_derivatives, output_scale)

        q0 = np.zeros((*self.ode_shape, num_derivatives + 1))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (a, noise), (p, p_inv)

        return discretise

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        cholesky_shape = (*self.ode_shape, num_derivatives + 1, num_derivatives + 1)
        cholesky = np.zeros(cholesky_shape)
        mean = np.stack(tcoeffs).T
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        A_new = p[None, :, None] * A * p_inv[None, None, :]
        noise = self.preconditioner_apply(noise, p)
        return _conditional.Conditional(A_new, noise)

    def standard_normal(self, ndim, output_scale):
        mean = np.zeros((*self.ode_shape, ndim))
        cholesky = output_scale[:, None, None] * np.eye(ndim)[None, ...]
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        if np.ndim(mean) > 0:
            assert np.shape(mean) == np.shape(x)
            return functools.vmap(self.update_mean, in_axes=(0, 0, None))(mean, x, num)

        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)


def system_matrices_1d(num_derivatives, output_scale):
    """Construct the IBM system matrices."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = np.flip(_hilbert(x))
    return A_1d, output_scale * linalg.cholesky_factor(Q_1d)


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


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = functools.vmap(k, in_axes=(0, None), out_axes=-1)
    return functools.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))
