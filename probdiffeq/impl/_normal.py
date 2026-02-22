from probdiffeq.backend import abc, func, linalg, np, tree
from probdiffeq.backend.typing import Callable, Sequence, TypeVar

T = TypeVar("T")
C = TypeVar("C", bound=Sequence)


class Normal:
    def __init__(self, mean):
        self.mean = mean

    @classmethod
    def from_dirac(cls, mean, *, damp):
        raise NotImplementedError

    def eval_mean(self):
        raise NotImplementedError

    def eval_standard_deviation(self):
        raise NotImplementedError

    def mahalanobis_norm_relative(self, u):
        raise NotImplementedError

    # Useless below

    # def logpdf(self, u, /, rv):
    #     raise NotImplementedError

    # def mean(self, rv):
    #     raise NotImplementedError

    # def hidden_shape(self, rv):
    #     raise NotImplementedError

    # def transform_unit_sample(self, unit_sample, /, rv):
    #     raise NotImplementedError

    # def to_multivariate_normal(self, u, rv):
    #     raise NotImplementedError

    # def rescale_cholesky(self, factor, /):
    #     raise NotImplementedError

    # def qoi(self, rv):
    #     raise NotImplementedError

    # def qoi_from_sample(self, sample, /):
    #     raise NotImplementedError

    # def update_mean(self, mean, x, /, num):
    #     raise NotImplementedError


class NormalDense(Normal):
    def __init__(self, mean: T, cholesky: T, unravel: Callable[[T], C]):
        super().__init__(mean=mean)
        self.cholesky = cholesky
        self.unravel = unravel

    @classmethod
    def from_dirac(cls, mean, *, damp):
        mean_flat, unravel = tree.ravel_pytree(mean)
        (d,) = mean_flat.shape
        cholesky = np.eye(d) * damp
        return cls(mean=mean_flat, cholesky=cholesky, unravel=unravel)

    def eval_mean(self):
        if self.mean.ndim > 1:
            return func.vmap(NormalDense.eval_mean)(self)

        return self.unravel(self.mean)

    def eval_standard_deviation(self):
        if self.mean.ndim > 1:
            return func.vmap(NormalDense.eval_standard_deviation)(self)

        diag = np.einsum("ij,ij->i", self.cholesky, self.cholesky)
        std = np.sqrt(diag)
        return self.unravel(std)

    def mahalanobis_norm_relative(self, u):
        dx = u - self.mean
        residual_white = linalg.solve_triangular(self.cholesky.T, dx, trans="T")
        mahalanobis = linalg.qr_r(residual_white[:, None])
        return np.reshape(np.abs(mahalanobis) / np.sqrt(self.mean.size), ())

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky
        return NormalDense(self.mean, cholesky, unravel=self.unravel)

    @staticmethod
    def register_pytree_node():
        def flatten(normal):
            children = normal.mean, normal.cholesky
            aux = (normal.unravel,)
            return children, aux

        def unflatten(aux, children):
            (unravel,) = aux
            mean, cholesky = children
            return NormalDense(mean, cholesky, unravel)

        tree.register_pytree_node(NormalDense, flatten, unflatten)


NormalDense.register_pytree_node()


class NormalIso(Normal):
    def __init__(self, mean: T, cholesky: T, treedef):
        super().__init__(mean=mean)
        self.cholesky = cholesky
        self.treedef = treedef

    @classmethod
    def from_dirac(cls, mean, *, damp):
        leaves, structure = tree.tree_flatten(mean)
        mean_array = np.stack(leaves)
        n, _ = mean_array.shape
        cholesky = np.eye(n) * damp
        return cls(mean=mean_array, cholesky=cholesky, treedef=structure)

    def eval_mean(self):
        if self.mean.ndim > 2:
            return func.vmap(NormalIso.eval_mean)(self)

        return tree.tree_unflatten(self.treedef, [*self.mean])

    def eval_standard_deviation(self):
        if self.mean.ndim > 2:
            return func.vmap(NormalIso.eval_standard_deviation)(self)
        diag = np.einsum("ij,ji->i", self.cholesky, self.cholesky)
        std = np.sqrt(diag)
        return tree.tree_unflatten(self.treedef, [*std])

    def mahalanobis_norm_relative(self, u):
        if self.cholesky.size > 1:
            raise ValueError
        residual_white = (self.mean - u) / self.cholesky
        residual_white_matrix = linalg.qr_r(residual_white.T)
        return np.reshape(np.abs(residual_white_matrix) / np.sqrt(self.mean.size), ())

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky
        return NormalIso(self.mean, cholesky, treedef=self.treedef)

    @staticmethod
    def register_pytree_node():
        def flatten(normal):
            children = normal.mean, normal.cholesky
            aux = (normal.treedef,)
            return children, aux

        def unflatten(aux, children):
            (treedef,) = aux
            mean, cholesky = children
            return NormalIso(mean, cholesky, treedef)

        tree.register_pytree_node(NormalIso, flatten, unflatten)


NormalIso.register_pytree_node()


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
            scale = tree.tree_map(np.ones_like, loc)

        loc_flat, unravel = tree.ravel_pytree(loc)
        scale_flat, _ = tree.ravel_pytree(scale)
        assert loc_flat.shape == scale_flat.shape

        (ode_dim,) = self.ode_shape
        num_coeffs = len(loc)
        assert loc_flat.size == num_coeffs * ode_dim

        cholesky_flat = linalg.diagonal_matrix(scale_flat)

        # Register "unravel" as a Pytree so that we can
        # JIT functions around normals.
        # TODO: this is only temporary?
        # unravel = tree.Partial(unravel)
        return NormalDense(loc_flat, cholesky_flat, unravel=unravel)

    def preconditioner_apply(self, rv, p, /):
        mean = p * rv.mean
        cholesky = p[:, None] * rv.cholesky
        return NormalDense(mean, cholesky)

    def standard(self, ndim, /, output_scale):
        eye_n = np.eye(ndim)
        eye_d = output_scale * np.eye(*self.ode_shape)
        cholesky = np.kron(eye_d, eye_n)
        mean = np.zeros((*self.ode_shape, ndim)).reshape((-1,), order="F")
        return NormalDense(mean, cholesky)


class IsotropicNormal(NormalBackend):
    def __init__(self, ode_shape, tree_structure):
        self.ode_shape = ode_shape
        self.tree_structure = tree_structure

    def from_tcoeffs(self, loc: C, scale: C | None = None):
        if scale is None:
            scale = tree.tree_map(lambda _: np.ones(()), loc)

        def ravel(s):
            return tree.ravel_pytree(s)[0]

        loc_leaves, _ = tree.tree_flatten(loc)
        leaves_flat = tree.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(leaves_flat)

        scale_leaves, _ = tree.tree_flatten(scale)
        scale_flat = np.stack(scale_leaves)

        num_coeffs = len(loc)
        if scale_flat.shape != (num_coeffs,):
            msg = "'scale' must have the same pytree structure as loc, "
            msg += "but each leaf must be a scalar instead of an array"
            msg += f"Received: {scale}"
            raise ValueError(msg)

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return NormalIso(loc_flat, cholesky_flat, treedef=self.tree_structure)

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
            scale = tree.tree_map(np.ones_like, loc)

        def ravel(s):
            return tree.ravel_pytree(s)[0]

        # Flatten and reshape the mean
        loc_leaves, _ = tree.tree_flatten(loc)
        loc_leaves_flat = tree.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(loc_leaves_flat).T

        # Flatten and reshape the standard deviation
        scale_leaves, _ = tree.tree_flatten(scale)
        scale_leaves_flat = tree.tree_map(ravel, scale_leaves)
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
