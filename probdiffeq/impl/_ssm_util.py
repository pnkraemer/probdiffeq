"""SSM utilities."""

from probdiffeq.backend import abc
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util, cond_util, ibm_util


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

    # TODO: move those to random.py and cond.py instead?

    @abc.abstractmethod
    def identity_conditional(self, num_derivatives_per_ode_dimension, /):
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
        return cond_util.Conditional(A, noise)

    def ibm_transitions(self, num_derivatives, output_scale=1.0):
        a, q_sqrtm = ibm_util.system_matrices_1d(num_derivatives, output_scale)
        q0 = np.zeros((num_derivatives + 1,))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return cond_util.Conditional(a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        transition = np.eye(ndim)
        mean = np.zeros((ndim,))
        cov_sqrtm = np.zeros((ndim, ndim))
        noise = _normal.Normal(mean, cov_sqrtm)
        return cond_util.Conditional(transition, noise)

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
        a, q_sqrtm = ibm_util.system_matrices_1d(num_derivatives, output_scale)
        (d,) = self.ode_shape
        eye_d = np.eye(d)
        A = np.kron(eye_d, a)
        Q = np.kron(eye_d, q_sqrtm)

        ndim = d * (num_derivatives + 1)
        q0 = np.zeros((ndim,))
        noise = _normal.Normal(q0, Q)

        precon_fun = ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            p = np.tile(p, d)
            p_inv = np.tile(p_inv, d)
            return cond_util.Conditional(A, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        (d,) = self.ode_shape
        n = ndim * d

        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        return cond_util.Conditional(A, _normal.Normal(m, C))

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
        return cond_util.Conditional(A, noise)

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
        A, q_sqrtm = ibm_util.system_matrices_1d(num_derivatives, output_scale)
        q0 = np.zeros((num_derivatives + 1, *self.ode_shape))
        noise = _normal.Normal(q0, q_sqrtm)
        precon_fun = ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return cond_util.Conditional(A, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, num_hidden_states_per_ode_dim, /):
        m0 = np.zeros((num_hidden_states_per_ode_dim, *self.ode_shape))
        c0 = np.zeros((num_hidden_states_per_ode_dim, num_hidden_states_per_ode_dim))
        noise = _normal.Normal(m0, c0)
        matrix = np.eye(num_hidden_states_per_ode_dim)
        return cond_util.Conditional(matrix, noise)

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
        return cond_util.Conditional(A_new, noise)

    def standard_normal(self, num, /, output_scale):
        mean = np.zeros((num, *self.ode_shape))
        cholesky = output_scale * np.eye(num)
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)
