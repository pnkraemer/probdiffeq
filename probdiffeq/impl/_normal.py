from probdiffeq.backend import abc, containers
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array


class Normal(containers.NamedTuple):
    mean: Array
    cholesky: Array


class NormalBackend(abc.ABC):
    @abc.abstractmethod
    def from_tcoeffs(self, tcoeffs, /, num_derivatives) -> Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, rv: Normal, p, /) -> Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def standard(self, num, /, output_scale) -> Normal:
        raise NotImplementedError


class ScalarNormal(NormalBackend):
    def from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = np.stack(tcoeffs)
        m0_corrected = np.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = np.zeros((num_derivatives + 1, num_derivatives + 1))
        return Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return Normal(p * rv.mean, p[:, None] * rv.cholesky)

    def standard(self, ndim, /, output_scale):
        mean = np.zeros((ndim,))
        cholesky = output_scale * np.eye(ndim)
        return Normal(mean, cholesky)


class DenseNormal(NormalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def from_tcoeffs(self, tcoeffs, /, num_derivatives):
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

        return Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        mean = p * rv.mean
        cholesky = p[:, None] * rv.cholesky
        return Normal(mean, cholesky)

    def standard(self, ndim, /, output_scale):
        eye_n = np.eye(ndim)
        eye_d = output_scale * np.eye(*self.ode_shape)
        cholesky = np.kron(eye_d, eye_n)
        mean = np.zeros((*self.ode_shape, ndim)).reshape((-1,), order="F")
        return Normal(mean, cholesky)


class IsotropicNormal(NormalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = f"The number of Taylor coefficients {len(tcoeffs)} does not match "
            msg2 = f"the number of derivatives {num_derivatives+1} in the solver."
            raise ValueError(msg1 + msg2)

        c_sqrtm0_corrected = np.zeros((num_derivatives + 1, num_derivatives + 1))
        m0_corrected = np.stack(tcoeffs)
        return Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return Normal(p[:, None] * rv.mean, p[:, None] * rv.cholesky)

    def standard(self, num, /, output_scale):
        mean = np.zeros((num, *self.ode_shape))
        cholesky = output_scale * np.eye(num)
        return Normal(mean, cholesky)


class BlockDiagNormal(NormalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        cholesky_shape = (*self.ode_shape, num_derivatives + 1, num_derivatives + 1)
        cholesky = np.zeros(cholesky_shape)
        mean = np.stack(tcoeffs).T
        return Normal(mean, cholesky)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return Normal(mean, cholesky)

    def standard(self, ndim, output_scale):
        mean = np.zeros((*self.ode_shape, ndim))
        cholesky = output_scale[:, None, None] * np.eye(ndim)[None, ...]
        return Normal(mean, cholesky)
