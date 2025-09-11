from probdiffeq.backend import abc, containers, linalg, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array


@tree_util.register_dataclass
@containers.dataclass
class Normal:
    mean: Array
    cholesky: Array


class NormalBackend(abc.ABC):
    @abc.abstractmethod
    def from_tcoeffs(self, tcoeffs: list, damp: float = 0.0):
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

    def from_tcoeffs(self, tcoeffs: list, damp: float = 0.0):
        m0_corrected, _ = tree_util.ravel_pytree(tcoeffs)

        (ode_dim,) = self.ode_shape
        ndim = len(tcoeffs)
        powers = 1 / np.arange(1, ndim + 1)
        powers = np.repeat(powers, ode_dim)
        cholesky = linalg.diagonal_matrix(damp**powers)

        return Normal(m0_corrected, cholesky)

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

    def from_tcoeffs(self, tcoeffs: list, damp: float = 0.0):
        powers = 1 / np.arange(1, len(tcoeffs) + 1)
        c_sqrtm0_corrected = linalg.diagonal_matrix(damp**powers)

        leaves, _ = tree_util.tree_flatten(tcoeffs)
        leaves_flat = tree_util.tree_map(lambda s: tree_util.ravel_pytree(s)[0], leaves)
        m0_corrected = np.stack(leaves_flat)
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

    def from_tcoeffs(self, tcoeffs: list, damp: float = 0.0):
        powers = 1 / np.arange(1, len(tcoeffs) + 1)
        cholesky = linalg.diagonal_matrix(damp**powers)
        cholesky = np.ones((*self.ode_shape, 1, 1)) * cholesky[None, ...]

        leaves, _ = tree_util.tree_flatten(tcoeffs)
        leaves_flat = tree_util.tree_map(lambda s: tree_util.ravel_pytree(s)[0], leaves)
        mean = np.stack(leaves_flat).T
        return Normal(mean, cholesky)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return Normal(mean, cholesky)

    def standard(self, ndim, output_scale):
        mean = np.zeros((*self.ode_shape, ndim))
        cholesky = output_scale[:, None, None] * np.eye(ndim)[None, ...]
        return Normal(mean, cholesky)
