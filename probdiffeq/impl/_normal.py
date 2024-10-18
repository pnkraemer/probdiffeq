from probdiffeq.backend import abc, containers, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array


class Normal(containers.NamedTuple):
    mean: Array
    cholesky: Array


class NormalBackend(abc.ABC):
    @abc.abstractmethod
    def from_tcoeffs(self, tcoeffs: list):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, rv: Normal, p, /) -> Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def standard(self, num, /, output_scale) -> Normal:
        raise NotImplementedError


class DenseNormal(NormalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def from_tcoeffs(self, tcoeffs: list):
        if tcoeffs[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_corrected, _ = tree_util.ravel_pytree(tcoeffs)

        (ode_dim,) = self.ode_shape
        ndim = len(tcoeffs) * ode_dim
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

    def from_tcoeffs(self, tcoeffs: list):
        c_sqrtm0_corrected = np.zeros((len(tcoeffs), len(tcoeffs)))
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

    def from_tcoeffs(self, tcoeffs: list):
        cholesky_shape = (*self.ode_shape, len(tcoeffs), len(tcoeffs))
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
