from probdiffeq.backend import func, linalg, np, random, tree
from probdiffeq.backend.typing import Callable, Sequence, TypeVar
from probdiffeq.util import cholesky_util

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

    @classmethod
    def from_tcoeffs(cls, loc: C, scale: C | None = None):
        raise RuntimeError

    @classmethod
    def from_mean_and_std(cls, mean, std):
        mean_flat, unravel = tree.ravel_pytree(mean)
        std_flat, _unravel = tree.ravel_pytree(std)

        assert mean_flat.shape == std_flat.shape
        cholesky = linalg.diagonal_matrix(std_flat)
        return cls(mean=mean_flat, cholesky=cholesky, unravel=unravel)

    @classmethod
    def from_standard(cls, ndim, /, output_scale):
        raise RuntimeError
        eye_n = np.eye(ndim)
        eye_d = output_scale * np.eye(*self.ode_shape)
        cholesky = np.kron(eye_d, eye_n)
        mean = np.zeros((*self.ode_shape, ndim)).reshape((-1,), order="F")
        return cls(mean, cholesky)

    def __repr__(self):
        return f"NormalDense(mean={self.mean}, cholesky={self.cholesky}, unravel={self.unravel})"

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

    def logpdf(self, u, /):
        u, _ = tree.ravel_pytree(u)
        cholesky = linalg.qr_r(self.cholesky.T).T
        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))

        dx = u - self.mean
        residual_white = linalg.solve_triangular(cholesky, dx, lower=True, trans=0)
        sqrnorm = linalg.vector_dot(residual_white, residual_white)

        const = np.log(np.pi() * 2)
        return -1 / 2 * sqrnorm - u.size / 2 * const - slogdet

    def to_multivariate_normal(self):
        return self.mean, self.cholesky @ self.cholesky.T

    def sample(self, key):
        base = random.normal(key, shape=self.mean.shape)
        sample_latent = self.mean + self.cholesky @ base
        return self.unravel(sample_latent)

    @staticmethod
    def update_moving_avg(mean, x, /, num):
        nominator = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        denominator = np.sqrt(num + 1)
        return nominator / denominator

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


class NormalIso(Normal):
    def __init__(self, mean: T, cholesky: T, treedef):
        super().__init__(mean=mean)
        self.cholesky = cholesky
        self.treedef = treedef

    def __repr__(self):
        return f"NormalIso(mean={self.mean}, cholesky={self.cholesky}, treedef={self.treedef})"

    @classmethod
    def from_dirac(cls, mean, *, damp):
        leaves, structure = tree.tree_flatten(mean)
        mean_array = np.stack(leaves)
        n, _ = mean_array.shape
        cholesky = np.eye(n) * damp
        return cls(mean=mean_array, cholesky=cholesky, treedef=structure)

    @classmethod
    def from_tcoeffs(cls, loc: C, scale: C | None = None):
        raise RuntimeError

    @classmethod
    def from_mean_and_std(cls, mean, std):

        def ravel(s):
            return tree.ravel_pytree(s)[0]

        loc_leaves, treedef = tree.tree_flatten(mean)
        leaves_flat = tree.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(leaves_flat)

        scale_leaves, _ = tree.tree_flatten(std)
        scale_flat = np.stack(scale_leaves)

        num_coeffs = len(mean)
        if scale_flat.shape != (num_coeffs,):
            msg = "'scale' must have the same pytree structure as loc, "
            msg += "but each leaf must be a scalar instead of an array"
            msg += f"Received: {scale}"
            raise ValueError(msg)

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return cls(loc_flat, cholesky_flat, treedef=treedef)

    @classmethod
    def from_standard(cls, num, /, output_scale):
        raise RuntimeError
        mean = np.zeros((num, *self.ode_shape))
        cholesky = output_scale * np.eye(num)
        return Normal(mean, cholesky)

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

    def logpdf(self, u, /):
        u_leaves = tree.tree_leaves(u)
        u_flat = tree.tree_map(lambda s: tree.ravel_pytree(s)[0], u_leaves)
        u_latent = np.stack(u_flat)

        # # if the gain is qoi-to-hidden, the data is a (d,) array.
        # # this is problematic for the isotropic model unless we explicitly broadcast.
        # if np.ndim(u) == 1:
        #     u = u[None, :]

        # Batch in the "mean" dimension and sum the results.
        rv_batch = NormalIso(1, None, treedef=self.treedef)
        logpdf_vmap = func.vmap(NormalIso.logpdf_scalar, in_axes=(rv_batch, 1))
        logpdfs = logpdf_vmap(self, u_latent)
        return np.sum(logpdfs)

    def logpdf_scalar(self, u, /):
        cholesky = linalg.qr_r(self.cholesky.T).T

        dx = u - self.mean
        w = linalg.solve_triangular(cholesky.T, dx, trans="T")

        maha_term = linalg.vector_dot(w, w)

        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * np.log(np.pi() * 2))

    def to_multivariate_normal(self):
        ode_state = tree.tree_unflatten(self.treedef, self.mean)[0]
        ode_shape = tree.ravel_pytree(ode_state)[0].shape
        eye_d = np.eye(*ode_shape)

        cov = self.cholesky @ self.cholesky.T

        cov = np.kron(eye_d, cov)
        mean = self.mean.reshape((-1,), order="F")
        return (mean, cov)

    def sample(self, key):
        n, _n = self.cholesky.shape

        base = random.normal(key, shape=(n,))
        sample_latent = self.mean + (self.cholesky @ base)[:, None]

        return tree.tree_unflatten(self.treedef, [*sample_latent])

    @staticmethod
    def update_moving_avg(mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)

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


class NormalBlockDiag(Normal):
    def __init__(self, mean, cholesky, treedef, unravel_leaf):
        super().__init__(mean=mean)
        self.cholesky = cholesky
        self.treedef = treedef
        self.unravel_leaf = unravel_leaf

    def __repr__(self):
        return f"NormalBlockDiag(mean={self.mean}, cholesky={self.cholesky}, treedef={self.treedef}, unravel_leaf=<function>)"

    @classmethod
    def from_dirac(cls, mean, *, damp):
        leaves, treedef = tree.tree_flatten(mean)
        leaves_flat = tree.tree_map(lambda s: tree.ravel_pytree(s)[0], leaves)
        flat = np.stack(leaves_flat).T
        _, unravel_leaf = tree.ravel_pytree(leaves[0])

        mean_array = flat
        d, n = np.shape(mean_array)
        cholesky = np.ones((d, 1, 1)) * (np.eye(n) * damp)[None, :, :]
        return cls(mean_array, cholesky, treedef=treedef, unravel_leaf=unravel_leaf)

    @classmethod
    def from_tcoeffs(cls, loc: C, scale: C | None = None):
        raise RuntimeError

    @classmethod
    def from_mean_and_std(cls, mean, std):
        def ravel(s):
            return tree.ravel_pytree(s)[0]

        # Flatten and reshape the mean
        loc_leaves, treedef = tree.tree_flatten(mean)
        loc_leaves_flat = tree.tree_map(ravel, loc_leaves)
        loc_flat = np.stack(loc_leaves_flat).T
        _, unravel_leaf = tree.ravel_pytree(loc_leaves[0])

        def unravel(z):
            z1 = tree.tree_unflatten(treedef, z.T)
            return tree.tree_map(unravel_leaf, z1)

        # Flatten and reshape the standard deviation
        scale_leaves, _ = tree.tree_flatten(std)
        scale_leaves_flat = tree.tree_map(ravel, scale_leaves)
        scale_flat = np.stack(scale_leaves_flat).T

        # Promote std into covariance matrix and apply damping
        num_coeffs = len(mean)
        d = np.ones((num_coeffs,))
        cholesky = linalg.diagonal_matrix(d)
        cholesky_flat = scale_flat[..., None] * cholesky[None, ...]
        return cls(loc_flat, cholesky_flat, treedef=treedef, unravel_leaf=unravel_leaf)

    @classmethod
    def from_standard(self, ndim, output_scale):
        raise RuntimeError
        mean = np.zeros((*self.ode_shape, ndim))
        cholesky = output_scale[:, None, None] * np.eye(ndim)[None, ...]
        return Normal(mean, cholesky)

    def eval_mean(self):
        if self.mean.ndim > 2:
            return func.vmap(NormalBlockDiag.eval_mean)(self)
        mean_tree = tree.tree_unflatten(self.treedef, [*(self.mean.T)])
        return tree.tree_map(self.unravel_leaf, mean_tree)

    def eval_standard_deviation(self):
        if self.mean.ndim > 2:
            return func.vmap(NormalBlockDiag.eval_standard_deviation)(self)
        diag = np.einsum("ijk,ikj->ij", self.cholesky, self.cholesky)
        std = np.sqrt(diag)
        std_tree = tree.tree_unflatten(self.treedef, [*(std.T)])
        return tree.tree_map(self.unravel_leaf, std_tree)

    def mahalanobis_norm_relative(self, u, /):
        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        mean = np.reshape(self.mean, (-1,))
        cholesky = np.reshape(self.cholesky, (-1,))
        return (mean - u) / cholesky / np.sqrt(mean.size)

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky
        return NormalBlockDiag(
            self.mean, cholesky, treedef=self.treedef, unravel_leaf=self.unravel_leaf
        )

    def logpdf(self, u, /):
        u_leaves = tree.tree_leaves(u)
        u_flat = tree.tree_map(lambda s: tree.ravel_pytree(s)[0], u_leaves)
        u_ = np.stack(u_flat).T
        return np.sum(func.vmap(NormalBlockDiag.logpdf_scalar)(self, u_))

    def logpdf_scalar(self, u):
        cholesky = linalg.qr_r(self.cholesky.T).T

        dx = u - self.mean
        w = linalg.solve_triangular(cholesky.T, dx, trans="T")

        maha_term = linalg.vector_dot(w, w)

        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * np.log(np.pi() * 2))

    def to_multivariate_normal(self):
        mean = np.reshape(self.mean.T, (-1,), order="F")
        cov = np.block_diag(self._cov_dense())
        return mean, cov

    def _cov_dense(self):
        if self.cholesky.ndim > 2:
            return func.vmap(NormalBlockDiag._cov_dense)(self)
        return self.cholesky @ self.cholesky.T

    def sample(self, key):
        if self.cholesky.ndim > 3:
            d, *_ = self.cholesky.shape
            keys = random.split(key, num=d)
            return func.vmap(NormalBlockDiag.sample)(self, keys)

        d, _n, n = self.cholesky.shape
        base = random.normal(key, shape=(d, n))
        sample_latent = self.mean + np.einsum("ijk,ij->ik", self.cholesky, base)

        tree_sample = tree.tree_unflatten(self.treedef, [*(sample_latent.T)])
        return tree.tree_map(self.unravel_leaf, tree_sample)

    @staticmethod
    def update_moving_avg(mean, x, /, num):
        if np.ndim(mean) > 0:
            assert np.shape(mean) == np.shape(x)
            fun = NormalBlockDiag.update_moving_avg
            return func.vmap(fun, in_axes=(0, 0, None))(mean, x, num)

        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)

    @staticmethod
    def register_pytree_node():
        def flatten(normal):
            children = normal.mean, normal.cholesky
            aux = (normal.treedef, normal.unravel_leaf)
            return children, aux

        def unflatten(aux, children):
            (treedef, unravel_leaf) = aux
            mean, cholesky = children
            return NormalBlockDiag(mean, cholesky, treedef, unravel_leaf)

        tree.register_pytree_node(NormalBlockDiag, flatten, unflatten)


class BlockDiagNormal:
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape


NormalDense.register_pytree_node()
NormalIso.register_pytree_node()
NormalBlockDiag.register_pytree_node()
