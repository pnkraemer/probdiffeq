from probdiffeq.backend import abc, containers, linalg, np, tree_util
from probdiffeq.backend.typing import Generic, Sequence, TypeVar

T = TypeVar("T")


@tree_util.register_dataclass
@containers.dataclass
class Normal(Generic[T]):
    mean: T
    cholesky: T


C = TypeVar("C", bound=Sequence)


class NormalBackend(abc.ABC):
    @abc.abstractmethod
    def from_tcoeffs(self, loc: C, scale: C | None = None):
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

    def from_tcoeffs(self, loc: C, scale: C | None = None):
        if scale is None:
            scale = tree_util.tree_map(np.ones_like, loc)

        loc_flat, _ = tree_util.ravel_pytree(loc)
        scale_flat, _ = tree_util.ravel_pytree(scale)
        assert loc_flat.shape == scale_flat.shape

        (ode_dim,) = self.ode_shape
        num_coeffs = len(loc)
        assert loc_flat.size == num_coeffs * ode_dim

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return Normal(loc_flat, cholesky_flat)

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

    def from_tcoeffs(self, loc: C, scale: C | None = None):
        if scale is None:
            scale = tree_util.tree_map(lambda _: np.ones(()), loc)

        def ravel(s):
            return tree_util.ravel_pytree(s)[0]

        loc_leaves, _ = tree_util.tree_flatten(loc)
        leaves_flat = tree_util.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(leaves_flat)

        scale_leaves, _ = tree_util.tree_flatten(scale)
        scale_flat = np.stack(scale_leaves)

        num_coeffs = len(loc)
        if scale_flat.shape != (num_coeffs,):
            msg = "'scale' must have the same pytree structure as loc, "
            msg += "but each leaf must be a scalar instead of an array"
            msg += f"Received: {scale}"
            raise ValueError(msg)

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return Normal(loc_flat, cholesky_flat)

    def preconditioner_apply(self, rv, p, /):
        return Normal(p[:, None] * rv.mean, p[:, None] * rv.cholesky)

    def standard(self, num, /, output_scale):
        mean = np.zeros((num, *self.ode_shape))
        cholesky = output_scale * np.eye(num)
        return Normal(mean, cholesky)


class BlockDiagNormal(NormalBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def from_tcoeffs(self, loc: C, scale: C | None = None):
        if scale is None:
            scale = tree_util.tree_map(np.ones_like, loc)

        def ravel(s):
            return tree_util.ravel_pytree(s)[0]

        # Flatten and reshape the mean
        loc_leaves, _ = tree_util.tree_flatten(loc)
        loc_leaves_flat = tree_util.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(loc_leaves_flat).T

        # Flatten and reshape the standard deviation
        scale_leaves, _ = tree_util.tree_flatten(scale)
        scale_leaves_flat = tree_util.tree_map(ravel, scale_leaves)
        scale_flat = np.stack(scale_leaves_flat).T

        # Promote std into covariance matrix and apply damping
        num_coeffs = len(loc)
        d = np.ones((num_coeffs,))
        cholesky = linalg.diagonal_matrix(d)
        cholesky_flat = scale_flat[..., None] * cholesky[None, ...]
        return Normal(loc_flat, cholesky_flat)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return Normal(mean, cholesky)

    def standard(self, ndim, output_scale):
        mean = np.zeros((*self.ode_shape, ndim))
        cholesky = output_scale[:, None, None] * np.eye(ndim)[None, ...]
        return Normal(mean, cholesky)
