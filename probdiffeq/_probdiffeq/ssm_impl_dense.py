from probdiffeq._probdiffeq import problems, ssm_impl_api, taylor_points, utilities
from probdiffeq.backend import func, linalg, np, random, structs, tree, warnings
from probdiffeq.backend.typing import Array, Callable, Sequence, TypeVar
from probdiffeq.util import cholesky_util, gram_util

__all__ = ["state_space_model_dense"]

C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""


@structs.dataclass
class DenseTreeFlatten(ssm_impl_api.AbstractTreeFlatten):
    """Implementation of flattening information for dense models."""

    unravel: Callable

    def flatten_tree(self, x):
        return tree.ravel_pytree(x)[0]

    def unflatten_array(self, x):
        return self.unravel(x)

    @classmethod
    def from_example(cls, x):
        _, unravel = tree.ravel_pytree(x)
        return cls(unravel)


class DenseNormal(ssm_impl_api.AbstractTreeNormal[DenseTreeFlatten]):
    """Construct a dense implementation of a normal distribution."""

    @classmethod
    def from_dirac(cls, mean, *, damp):
        utilities.verify_taylor_coefficient_pytree(mean)
        std = tree.tree_map(lambda x: np.ones_like(x) * damp, mean)
        return DenseNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        utilities.verify_taylor_coefficient_pytree(mean)
        utilities.verify_taylor_coefficient_pytree(std)

        tree_flatten = DenseTreeFlatten.from_example(mean)

        mean_flat = tree_flatten.flatten_tree(mean)
        std_flat = tree_flatten.flatten_tree(std)

        assert mean_flat.shape == std_flat.shape
        cholesky = linalg.diagonal_matrix(std_flat)
        return cls(mean_flat, cholesky, tree_flatten)

    @property
    def batch_shape(self):
        *shape, _n = self.mean_flat.shape
        return shape

    @property
    def mean(self):
        return self._mean_batched()

    def _mean_batched(self):
        if self.mean_flat.ndim > 1:
            return func.vmap(DenseNormal._mean_batched)(self)
        return self.tree_flatten.unflatten_array(self.mean_flat)

    @property
    def std(self):
        return self._std_batched()

    def _std_batched(self):
        if self.mean_flat.ndim > 1:
            return func.vmap(DenseNormal._std_batched)(self)

        std_flat = func.vmap(linalg.qr_r)(self.cholesky_flat[..., None])
        std_flat = np.abs(std_flat.reshape((-1,)))
        return self.tree_flatten.unflatten_array(std_flat)

    def residual_whitened_rms_tree(self, u):
        u, _ = tree.ravel_pytree(u)
        return self.residual_whitened_rms_flat(u)

    def residual_whitened_rms_flat(self, u):
        dx = u - self.mean_flat
        residual_white = linalg.solve_triu(self.cholesky_flat.T, dx, trans="T")
        mahalanobis = linalg.qr_r(residual_white[:, None])
        return np.reshape(np.abs(mahalanobis) / np.sqrt(self.mean_flat.size), ())

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky_flat
        return DenseNormal(self.mean_flat, cholesky, self.tree_flatten)

    def logpdf_tree(self, u, /):
        u, _ = tree.ravel_pytree(u)
        return self.logpdf_flat(u)

    def logpdf_flat(self, u, /):
        cholesky = linalg.qr_r(self.cholesky_flat.T).T
        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))

        dx = u - self.mean_flat
        residual_white = linalg.solve_tril(cholesky, dx, trans=0)
        sqrnorm = linalg.vector_dot(residual_white, residual_white)

        const = np.log(np.pi() * 2)
        return -1 / 2 * sqrnorm - u.size / 2 * const - slogdet

    def to_multivariate_normal(self):
        if self.mean_flat.ndim > 1:
            return func.vmap(DenseNormal.to_multivariate_normal)(self)

        return self.mean_flat, self.cholesky_flat @ self.cholesky_flat.T

    def sample_tree(self, key):
        sample_flat = self.sample_flat(key)
        return self.tree_flatten.unflatten_array(sample_flat)

    def sample_flat(self, key):
        base = random.normal(key, shape=self.mean_flat.shape)
        return self.mean_flat + self.cholesky_flat @ base

    def identity_conditional(self) -> "DenseLatentCond":
        (n,) = self.mean_flat.shape
        A = np.eye(n)
        noise = DenseNormal(np.zeros((n,)), np.zeros((n, n)), self.tree_flatten)
        return DenseLatentCond.from_linop_and_noise(A, noise)

    # TODO: move prototype_output_scale_calibrated to the prior?
    def prototype_output_scale_calibrated(self):
        return np.ones(())

    def to_derivative(self, i, std):
        all_flat, all_unravel = tree.ravel_pytree(self.mean)

        def select(a):
            return tree.ravel_pytree(all_unravel(a)[i])[0]

        x = np.zeros(all_flat.shape)
        linop = func.jacfwd(select)(x)

        data_like = all_unravel(x)[0]
        noise = DenseNormal.from_mean_and_std([data_like], [std])
        return DenseLatentCond.from_linop_and_noise(linop, noise)

    @staticmethod
    def register_pytree_node() -> None:
        def flatten(normal):
            children = normal.mean_flat, normal.cholesky_flat
            aux = (normal.tree_flatten,)
            return children, aux

        def unflatten(aux, children):
            (tree_flatten,) = aux
            mean, cholesky = children
            return DenseNormal(mean, cholesky, tree_flatten)

        tree.register_pytree_node(DenseNormal, flatten, unflatten)


DenseNormal.register_pytree_node()


class DenseMatrix(ssm_impl_api.AbstractLinOp):
    def __init__(self, *, matrix_ndmd):
        *_batch, n_out, d_out, n_in, d_in = matrix_ndmd.shape
        super().__init__(n_in=n_in, n_out=n_out, d_in=d_in, d_out=d_out)
        self.matrix_ndmd = matrix_ndmd

    def matvec_dndm(self, vec):
        raise NotImplementedError

    def matvec_ndmd(self, vec):
        raise NotImplementedError

    def matvec_flat(self, vec):
        vec_nd = vec.reshape((self.n_in, self.d_in))
        vec_nd = linalg.einsum("...ijkl,...kl->...ij", self.matrix_ndmd, vec_nd)
        return vec_nd.reshape((self.n_out * self.d_out,))

    @classmethod
    def from_flat(cls, matrix, /, *, n_in, n_out, d_in, d_out):
        matrix_ndmd = matrix.reshape((n_out, d_out, n_in, d_in))
        return cls(matrix_ndmd=matrix_ndmd)

    @property
    def precon_prototype(self):
        return np.ones((self.d_in * self.n_in,)), np.ones((self.d_out * self.n_out,))

    def materialize_ndmd(self):
        dummy = np.ones((self.d_in * self.n_in,))
        A_flat = func.jacfwd(self.matvec_flat)(dummy)
        return A_flat.reshape((self.n_out, self.d_out, self.n_in, self.d_in))

    @classmethod
    def _register_as_pytree(cls) -> None:
        """Register this class (or a subclass) as a JAX pytree."""

        def flatten(linop):
            children = (linop.matrix_ndmd,)
            return children, ()

        def unflatten(_aux, children):
            (matrix_ndmd,) = children
            return cls(matrix_ndmd=matrix_ndmd)

        tree.register_pytree_node(cls, flatten, unflatten)


DenseMatrix._register_as_pytree()


class DenseLatentCond(ssm_impl_api.AbstractLatentCond):
    """Dense (full-covariance) implementation of LatentCond operations."""

    def apply_flat(self, x, /):
        x = self.to_latent * x
        Ax = self.A.matvec_flat(x)
        mean = self.to_observed * (Ax + self.noise.mean_flat)
        cholesky = self.to_observed[:, None] * self.noise.cholesky_flat
        return DenseNormal(mean, cholesky, self.noise.tree_flatten)

    def marginalise(self, rv, /):
        mean = self.to_latent * rv.mean_flat
        cholesky = self.to_latent[:, None] * rv.cholesky_flat

        R_stack = (self.A.matmat_flat(cholesky).T, self.noise.cholesky_flat.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = self.to_observed * (self.A.matvec_flat(mean) + self.noise.mean_flat)
        cholesky_new = self.to_observed[:, None] * cholesky_new
        return DenseNormal(mean_new, cholesky_new, self.noise.tree_flatten)

    def merge(self, other: "DenseLatentCond", /) -> "DenseLatentCond":
        # self = cond1 (outer), other = cond2 (inner)
        T = self.to_latent * other.to_observed

        g = self.A @ (T[:, None] * other.A)
        xi = self.A @ (T * other.noise.mean_flat) + self.noise.mean_flat

        R1 = (self.A @ (T[:, None] * other.noise.cholesky_flat)).T
        R2 = self.noise.cholesky_flat.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        noise = DenseNormal(xi, Xi.T, self.noise.tree_flatten)
        return DenseLatentCond(
            g, noise, to_latent=other.to_latent, to_observed=self.to_observed
        )

    def revert(self, rv: "DenseNormal", /, *, solve_triu: Callable):
        noise = self.noise

        # Pull into latent space
        mean = self.to_latent * rv.mean_flat
        cholesky = self.to_latent[:, None] * rv.cholesky_flat

        # QR decomposition
        matmat = func.vmap(self.A.matvec_flat, in_axes=-1, out_axes=-1)
        R_X_F = matmat(cholesky).T
        R_X = cholesky.T
        R_YX = noise.cholesky_flat.T
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional(
            R_X_F=R_X_F, R_X=R_X, R_YX=R_YX, solve_triu=solve_triu
        )

        # Complete update from QR results (keep in latent space)
        mean_observed = self.A.matvec_flat(mean) + noise.mean_flat
        mean_corrected = mean - gain @ mean_observed
        cholesky_corrected = r_cor.T
        corrected = DenseNormal(mean_corrected, cholesky_corrected, rv.tree_flatten)

        # Save the conditional
        gain_ndmd = gain.reshape((self.A.n_in, self.A.d_in, self.A.n_out, self.A.d_out))
        cond_new = DenseLatentCond(
            DenseMatrix(matrix_ndmd=gain_ndmd),
            corrected,
            to_latent=1 / self.to_observed,  # is diagonal
            to_observed=1 / self.to_latent,  # is diagonal
        )

        # Push marginals into latent space and remove offdiagonals from the Cholesky
        mean = self.to_observed * mean_observed
        cholesky = self.to_observed[:, None] * r_obs.T
        observed = DenseNormal(mean, cholesky, noise.tree_flatten)
        return observed, cond_new

    def preconditioner_apply(self, /):
        A = self.to_observed[:, None] * self.A * self.to_latent[None, :]
        mean = self.to_observed * self.noise.mean_flat
        cholesky = self.to_observed[:, None] * self.noise.cholesky_flat
        noise = DenseNormal(mean, cholesky, self.noise.tree_flatten)
        return DenseLatentCond.from_linop_and_noise(A, noise)


DenseLatentCond._register_as_pytree()


class DenseLatentCondProjected(ssm_impl_api.AbstractLatentCondProjected):
    def apply_flat(self, x, /):
        dense_cond = DenseLatentCond(
            A=self.A,
            noise=self.noise,
            to_latent=self.to_latent,
            to_observed=self.to_observed,
        )
        return dense_cond.apply_flat(x)

    def marginalise(self, rv, /):
        dense_cond = DenseLatentCond(
            A=self.A,
            noise=self.noise,
            to_latent=self.to_latent,
            to_observed=self.to_observed,
        )
        observed = dense_cond.marginalise(rv)
        m = len(observed.mean)
        d = observed.mean_flat.size // n
        cholesky = self._remove_offdiag(observed.cholesky_flat, n=m, d=d)
        return DenseNormal(observed.mean_flat, cholesky, noise.tree_flatten)

    def merge(self, other: "DenseLatentCondProjected", /) -> "DenseLatentCondProjected":
        raise RuntimeError

    def revert(self, rv: DenseNormal, /, *, solve_triu: Callable):
        # Compute the full conditional
        dense_cond = DenseLatentCond(
            A=self.A,
            noise=self.noise,
            to_latent=self.to_latent,
            to_observed=self.to_observed,
        )
        observed, cond_new = dense_cond.revert(rv, solve_triu=solve_triu)

        # Remove off-block-diagonal entries
        key, subkey = random.split(self.key, num=2)

        # Posterior
        cond = cond_new.noise
        n = len(cond.mean)
        d = cond.mean_flat.size // n
        keys = random.split(subkey, num=self.num_probes)
        ensembles = func.vmap(cond.sample_flat)(keys)
        ensembles_nd = ensembles.reshape((self.num_probes, n, d))
        cholesky = self._remove_offdiag_from_ensembles(ensembles_nd)
        noise = DenseNormal(cond.mean_flat, cholesky, cond.tree_flatten)

        key, subkey = random.split(self.key, num=2)
        construct = DenseLatentCondProjected.from_linop_and_noise_and_stochtrace
        cond_new = construct(
            cond_new.A, noise=noise, key=subkey, num_probes=self.num_probes
        )

        # Reduce observed
        m = len(observed.mean)
        d = observed.mean_flat.size // m
        keys = random.split(key, num=self.num_probes)
        ensembles = func.vmap(observed.sample_flat)(keys)
        ensembles_md = ensembles.reshape((self.num_probes, m, d))
        cholesky = self._remove_offdiag_from_ensembles(ensembles_md)
        observed = DenseNormal(observed.mean_flat, cholesky, noise.tree_flatten)

        # Return values
        return observed, cond_new

    def preconditioner_apply(self, /):
        raise RuntimeError

    @staticmethod
    def _remove_offdiag_from_ensembles(ensembles_snd, bias: bool = False):
        # 'snd' encodes the shape (S, n, d), as opposed to (S, n*d).

        def ensemble_to_sample_cholesky(s):
            """Compute a sample Cholesky factor from ensembles."""
            num, _n = s.shape
            s = s / np.sqrt(num) if bias else s / np.sqrt(num - 1)
            return linalg.qr_r(s).T

        # The QR decomposition is why we assume S >= n,
        # so let's check it briefly:
        S, m, d = ensembles_snd.shape
        if m > S:
            msg = "The function requires at least as many ensembles as Taylor coefficients."
            msg += f" Received: S={S} < m={m}, which violates this assumption."
            raise ValueError(msg)

        # Center the ensembles
        ensembles_snd -= ensembles_snd.mean(axis=0, keepdims=True)

        # Assume ensembles are shape (S, n, d), so we batch along d,
        # but since we also want the output to be (d, n, n), the out_axes is 0.
        transform = func.vmap(ensemble_to_sample_cholesky, in_axes=-1, out_axes=0)
        cholesky = transform(ensembles_snd)

        # Reintroduce the diagonal: (d, n, n) -> (n, d, n, d)
        cholesky = linalg.einsum("dnm,dt->ndmt", cholesky, np.eye(d))

        # Flatten to conform with DenseNormal's hidden shape
        return cholesky.reshape((m * d, m * d))


DenseLatentCondProjected._register_as_pytree()


class DenseOdeTs0(ssm_impl_api.AbstractOde):
    """Dense ODE linearization via TS0 (zeroth-degree Taylor series: evaluate at the prior mean, no Jacobian)."""

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv: DenseNormal, state: None, *, damp: float, t):
        del state

        def derivative_selector(m: Array) -> Array:
            """Select the n-th derivative from the Taylor coefficient stack."""
            m0 = rv.tree_flatten.unflatten_array(m)[self.ode.num_tcoeffs_in_args]
            return tree.ravel_pytree(m0)[0]

        Ms = rv.mean

        fm = self.ode.vector_field(jet_coords=Ms[: self.ode.num_tcoeffs_in_args], t=t)
        fx = tree.tree_map(lambda s: -s, [fm])
        linop = func.jacrev(derivative_selector)(rv.mean_flat)
        noise = DenseNormal.from_dirac(fx, damp=damp)
        cond = DenseLatentCond.from_linop_and_noise(linop, noise)
        return cond, None


class DenseOdeTs1(ssm_impl_api.AbstractOde):
    """Dense ODE linearization via TS1 (first-degree Taylor series: evaluate the residual and its Jacobian at the linearization point)."""

    def init_linearization(self):
        return self.ode.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t: float):
        # Read n and d so that we can turn latent arrays into
        # (n, d) arrays, which Jacobians require
        m_tree = rv.mean
        n = len(m_tree)
        d = rv.mean_flat.size // n

        # Rewrite the vector field as one that maps
        # Arrays to arrays.
        # Maps (n, d) to (1, d) to conform the Jacobian API
        def fun(s: Array) -> Array:
            # Move to latent space
            s = np.reshape(s, (-1,))

            # Extract all tcoeffs
            jet_coords = rv.tree_flatten.unflatten_array(s)

            # Extract relevant tcoeffs ("jet-coordinates")
            jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

            # Evaluate the actual vector field
            fs0 = self.ode.vector_field(jet_coords=jet_coords, t=t)

            # Bring back into (m, d) form.
            return tree.ravel_pytree(fs0)[0][None, :]

        # Materialize the Jacobian
        m0 = rv.mean_flat.reshape((n, d))
        fx, J, state = self.ode.jacobian.materialize_dense(fun, m0, state)

        # Flatten fx and J correctly (from [m, d, n, d] to [md,nd])
        m, d = fx.shape
        fx = fx.reshape((m * d,))
        J = J.reshape((m * d, -1))

        # Bulletproof construction of selection operators via jacobian(slicing)
        # instead of instantiating identity matrices and selecting rows

        @func.jacfwd
        def projection_e1(s):
            s_tree = rv.tree_flatten.unflatten_array(s)
            return tree.ravel_pytree(s_tree[self.ode.num_tcoeffs_in_args])[0]

        # Complete the expressions for bias and linop
        fx = J @ rv.mean_flat - fx
        E1 = projection_e1(rv.mean_flat)
        linop = E1 - J

        # Flatten fx into the correct pytree structure
        f0 = DenseNormal.from_dirac([m_tree[self.ode.num_tcoeffs_in_args]], damp=damp)
        fx = f0.tree_flatten.unflatten_array(fx)

        # Collect all quantities and return
        noise = DenseNormal.from_dirac(fx, damp=damp)
        linop = DenseMatrix.from_flat(linop, n_out=m, d_out=d, n_in=n, d_in=d)
        cond = DenseLatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class DenseOdeTs1Projected(ssm_impl_api.AbstractOdeProjected):
    def init_linearization(self):
        jac = self.ode.jacobian.init_jacobian_handler()
        return self.key, jac

    def linearize(self, rv, state, *, damp: float, t: float):
        key, jacstate = state
        del state

        # Read n and d so that we can turn latent arrays into
        # (n, d) arrays, which Jacobians require
        m_tree = rv.mean
        n = len(m_tree)
        d = rv.mean_flat.size // n

        # Rewrite the vector field as one that maps
        # Arrays to arrays.
        # Maps (n, d) to (1, d) to conform the Jacobian API
        def fun(s: Array) -> Array:
            # Move to latent space
            s = np.reshape(s, (-1,))

            # Extract all tcoeffs
            jet_coords = rv.tree_flatten.unflatten_array(s)

            # Extract relevant tcoeffs ("jet-coordinates")
            jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

            # Evaluate the actual vector field
            fs0 = self.ode.vector_field(jet_coords=jet_coords, t=t)

            # Bring back into (m, d) form.
            return tree.ravel_pytree(fs0)[0][None, :]

        # Materialize the Jacobian
        m0 = rv.mean_flat.reshape((n, d))

        fx, J, jacstate = self.ode.jacobian.materialize_dense(fun, m0, jacstate)

        # Flatten fx and J correctly (from [m, d, n, d] to [md,nd])
        m, d = fx.shape
        fx = fx.reshape((m * d,))
        J = J.reshape((m * d, -1))

        # Bulletproof construction of selection operators via jacobian(slicing)
        # instead of instantiating identity matrices and selecting rows

        @func.jacfwd
        def projection_e1(s):
            s_tree = rv.tree_flatten.unflatten_array(s)
            return tree.ravel_pytree(s_tree[self.ode.num_tcoeffs_in_args])[0]

        # Complete the expressions for bias and linop
        fx = J @ rv.mean_flat - fx
        E1 = projection_e1(rv.mean_flat)
        linop = E1 - J

        # Flatten fx into the correct pytree structure
        f0 = DenseNormal.from_dirac([m_tree[self.ode.num_tcoeffs_in_args]], damp=damp)
        fx = f0.tree_flatten.unflatten_array(fx)

        # Collect all quantities and return
        noise = DenseNormal.from_dirac(fx, damp=damp)
        linop = DenseMatrix.from_flat(linop, n_out=m, d_out=d, n_in=n, d_in=d)

        key, subkey = random.split(key, num=2)
        construct = DenseLatentCondProjected.from_linop_and_noise_and_stochtrace
        cond = construct(linop, noise, key=subkey, num_probes=self.num_probes)

        return cond, (key, jacstate)


class DenseResidual(ssm_impl_api.AbstractResidual):
    """Construct a dense implementation of residual-TS1 linearization."""

    def __init__(self, residual, *, taylor_point) -> None:
        super().__init__(residual)
        self.taylor_point = taylor_point

    def init_linearization(self):
        return self.residual.jacobian.init_jacobian_handler()

    def constraint_flat(self, *, tree_flatten) -> Callable:
        """Evaluate a flattened version of the residual constraint."""

        def fun(m: Array, *, t):
            # Unravel the location and extract derivatives
            m_tree = tree_flatten.unflatten_array(m)
            relevant_tcoeffs = m_tree[: self.residual.num_tcoeffs_in_args]

            # Evaluate the residual
            residual_eval = self.residual.residual_function(
                jet_coords=relevant_tcoeffs, t=t
            )

            # Flatten the output so that the Jacobians are matrices, not Pytrees.
            return tree.ravel_pytree(residual_eval)[0]

        return fun

    def linearize(self, rv, state, *, damp: float, t):

        # Fix all arguments except the Array ones
        constraint_flat = self.constraint_flat(tree_flatten=rv.tree_flatten)

        # Get the linearization point (i.e., prior or posterior linearisation)
        xi = self.taylor_point(constraint_flat, rv, t=t)

        # Read n and d so that we can turn latent arrays into
        # (n, d) arrays, which Jacobians require
        m_tree = rv.mean
        n = len(m_tree)
        d = rv.mean_flat.size // n

        # Rewrite the constraint as one that maps 2d arrays to 2d arrays.
        # Here, write the fun as (d, 1) to (d', 1) because the Jacobian
        # logic requires 2d to 2d arrays. Since we materialize the full
        # Jacobian, everything is still fine.
        def fun(s: Array) -> Array:
            # Move from (n, d) shape into latent space: (-1,) shape
            s = np.reshape(s, (-1,))

            # Evaluate the actual constraint
            fs0 = constraint_flat(s, t=t)
            # Bring back into (m, d) form.
            return fs0[:, None]

        # Evaluate the linearization
        xi_2d = xi.reshape((-1, 1))
        fx, J, state = self.residual.jacobian.materialize_dense(fun, xi_2d, state)
        m, d = fx.shape
        fx = fx.reshape((m * d,))
        J = J.reshape((m * d, -1))
        fx = fx - J @ xi

        if J.shape[0] > J.shape[1]:
            msg = f"There are more constraints ({J.shape[0]}) than variables ({J.shape[1]})."
            msg += " This will likely cause an error in the conditioning."
            warnings.warn(msg, stacklevel=1)

        # Turn the linearization into a conditional
        noise = DenseNormal.from_dirac([fx], damp=damp)

        cond = DenseLatentCond.from_linop_and_noise(J, noise)
        return cond, state


class DenseWienerIntegrated(ssm_impl_api.AbstractPrior):
    def __init__(self, init, output_scale, *, d, A, Q, q0, tree_flatten, precon_fun):
        super().__init__(init, output_scale)
        self.d = d
        self.A = A
        self.Q = Q
        self.q0 = q0
        self.tree_flatten = tree_flatten
        self.precon_fun = precon_fun

    def transition(self, *, dt, output_scale):
        output_scale = np.asarray(output_scale)
        if output_scale.shape != ():
            msg = "The output-scale has the wrong shape."
            msg += f" Expected: {()}."
            msg += f" Received: {output_scale.shape}."
            raise ValueError(msg)

        p, p_inv = self.precon_fun(dt)
        p = np.repeat(p, self.d)
        p_inv = np.repeat(p_inv, self.d)

        # Q contains the base-output scale, so we only multiply with output_scale
        noise = DenseNormal(self.q0, output_scale * self.Q, self.tree_flatten)
        return DenseLatentCond(self.A, noise, to_latent=p_inv, to_observed=p)

    @staticmethod
    def register_pytree():
        def flatten(iwp):
            children = (iwp.init, iwp.output_scale, iwp.A, iwp.Q, iwp.q0)
            aux = (iwp.d, iwp.tree_flatten, iwp.precon_fun)
            return children, aux

        def unflatten(aux, children):
            d, tf, precon_fun = aux
            init, output_scale, A, Q, q0 = children
            return DenseWienerIntegrated(
                init,
                output_scale,
                d=d,
                A=A,
                Q=Q,
                q0=q0,
                tree_flatten=tf,
                precon_fun=precon_fun,
            )

        tree.register_pytree_node(DenseWienerIntegrated, flatten, unflatten)


DenseWienerIntegrated.register_pytree()


class DenseExponential(ssm_impl_api.AbstractPrior):
    def __init__(
        self, init, output_scale, A, B, *, d, q0, tree_flatten, precon_fun, exp_gram
    ):
        super().__init__(init, output_scale)
        self.A = A
        self.B = B
        self.d = d
        self.q0 = q0
        self.tree_flatten = tree_flatten
        self.precon_fun = precon_fun
        self.exp_gram = exp_gram

    def transition(self, *, dt: float, output_scale: Array):
        output_scale = np.asarray(output_scale)
        if output_scale.shape != ():
            msg = "The output-scale has the wrong shape."
            msg += f" Expected: {()}."
            msg += f" Received: {output_scale.shape}."
            raise ValueError(msg)

        p, p_inv = self.precon_fun(dt)
        p = np.repeat(p, self.d)
        p_inv = np.repeat(p_inv, self.d)

        A_p = dt * p_inv[:, None] * self.A * p[None, :]
        B_p = np.sqrt(dt) * p_inv[:, None] * self.B
        eA, L = self.exp_gram(A_p, B_p)

        # L already contains the output scale information (via B),
        # so we only multiply with the incoming output scale
        noise = DenseNormal(self.q0, output_scale * L, self.tree_flatten)

        return DenseLatentCond(eA, noise, to_latent=p_inv, to_observed=p)

    @staticmethod
    def register_pytree():
        def flatten(expo):
            children = (expo.init, expo.output_scale, expo.A, expo.B, expo.q0)
            aux = (expo.d, expo.tree_flatten, expo.precon_fun, expo.exp_gram)
            return children, aux

        def unflatten(aux, children):
            d, tf, precon_fun, exp_gram = aux
            init, output_scale, A, B, q0 = children
            return DenseExponential(
                init,
                output_scale,
                A,
                B,
                d=d,
                q0=q0,
                tree_flatten=tf,
                precon_fun=precon_fun,
                exp_gram=exp_gram,
            )

        tree.register_pytree_node(DenseExponential, flatten, unflatten)


DenseExponential.register_pytree()


class state_space_model_dense(ssm_impl_api.StateSpaceModel):
    """Dense (full-covariance) state-space model implementation."""

    def prior_wiener_integrated(
        self,
        tcoeffs_mean: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        tcoeffs_std = self._tcoeffs_standard_deviation(
            tcoeffs_mean, is_exact=is_exact, inexact_eps=inexact_eps
        )
        return self.prior_wiener_integrated_diffuse(
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    def prior_wiener_integrated_diffuse(
        self,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        if diffuse_derivatives > 0:
            tcoeffs_mean, tcoeffs_std = self._add_diffuse_derivatives(
                tcoeffs_mean,
                tcoeffs_std,
                diffuse_derivatives=diffuse_derivatives,
                diffuse_eps=diffuse_eps,
            )

        # Construct the initial variable from the mean and std
        init = DenseNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)
        single_flat, single_unravel = tree.ravel_pytree(tcoeffs_mean[0])
        Lambda = self._process_base_scale(output_scale, single_flat, single_unravel)

        num_derivatives = len(tcoeffs_mean) - 1
        (d,) = single_flat.shape
        a, q_sqrtm_1d = utilities.system_matrices_1d_iwp(num_derivatives)
        eye_d = np.eye(d)
        A = np.kron(a, eye_d)
        Q = np.kron(q_sqrtm_1d, Lambda)
        q0 = np.zeros(((num_derivatives + 1) * d,))
        tf = DenseTreeFlatten.from_example(tcoeffs_mean)
        precon_fun = utilities.preconditioner_taylor(num_derivatives)

        n = len(tcoeffs_mean)
        A = DenseMatrix.from_flat(A, n_in=n, n_out=n, d_in=d, d_out=d)
        return DenseWienerIntegrated(
            init, Lambda, d=d, A=A, Q=Q, q0=q0, tree_flatten=tf, precon_fun=precon_fun
        )

    def prior_exponential(
        self,
        ode: problems.JetOdeAutonomous,
        tcoeffs_mean: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        """Construct an exponential integrator prior."""
        if ode.num_tcoeffs_in_args != len(tcoeffs_mean):
            msg = f"""The exponential prior does not match the Taylor coefficients in the SSM.

        Concretely:

        - For two Taylor coefficients, we expect an ODE of order 2.
        - For three Taylor coefficients, we expect an ODE of order 3.
        - For four Taylor coefficients, we expect an ODE of order 4.

        and so on. The passed ODE has order **{ode.num_tcoeffs_in_args}**,
        whereas the state-space model includes **{len(tcoeffs_mean)}**
        Taylor coefficients.
        """
            raise TypeError(msg)
        tcoeffs_std = self._tcoeffs_standard_deviation(
            tcoeffs_mean, is_exact=is_exact, inexact_eps=inexact_eps
        )
        return self.prior_exponential_diffuse(
            ode,
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    def prior_exponential_diffuse(
        self,
        ode: problems.JetOdeAutonomous,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        if diffuse_derivatives > 0:
            tcoeffs_mean, tcoeffs_std = self._add_diffuse_derivatives(
                tcoeffs_mean,
                tcoeffs_std,
                diffuse_derivatives=diffuse_derivatives,
                diffuse_eps=diffuse_eps,
            )

        # Construct the initial variable from the mean and std
        init = DenseNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        # Process the base-scale
        single_flat, single_unravel = tree.ravel_pytree(tcoeffs_mean[0])
        Lambda = self._process_base_scale(output_scale, single_flat, single_unravel)

        # Turn the linear vector field into the bottom block of the IOUP
        # by building the matrix-version of the vector field's Jacobian.

        # First, set up a template variable

        leaves_flat, unflatten = tree.ravel_pytree(tcoeffs_mean)

        def vf_flat(tcoeffs_flat):
            tcoeffs_tree = unflatten(tcoeffs_flat)
            fx = ode.autonomous(jet_coords=tcoeffs_tree)
            return tree.ravel_pytree(fx)[0]

        bottom_block = func.jacfwd(vf_flat)(leaves_flat)

        # Construct the SDE matrices
        num_derivatives = len(tcoeffs_mean) - 1
        (d,) = single_flat.shape
        eye_d = np.eye(d)
        a = linalg.diagonal_matrix(np.ones((num_derivatives,)), k=1)
        A = np.kron(a, eye_d)
        A = A.at[-d:, :].set(bottom_block)

        b = np.eye(num_derivatives + 1)[-1][:, None]
        B = np.kron(b, Lambda)

        q0 = np.zeros((num_derivatives + 1) * d)
        tf = DenseTreeFlatten.from_example(tcoeffs_mean)
        precon_fun = utilities.preconditioner_taylor(num_derivatives)
        dtype_str = str(B.dtype)
        pade_legendre = (
            gram_util.pade_and_legendre_9()
            if dtype_str == "float64"
            else gram_util.pade_and_legendre_5()
        )
        exp_gram = gram_util.exp_gram_cholesky(
            pade_legendre=pade_legendre, solve=linalg.solve_lu
        )
        return DenseExponential(
            init,
            Lambda,
            A,
            B,
            d=d,
            q0=q0,
            tree_flatten=tf,
            precon_fun=precon_fun,
            exp_gram=exp_gram,
        )

    def _tcoeffs_standard_deviation(self, tcoeffs_mean, /, *, is_exact, inexact_eps):

        # Construct the initial std.
        # If is_exact is a boolean, copy the pytree structure from the mean
        # Otherwise, set the initial std element-wise.
        if isinstance(is_exact, bool):
            if is_exact:
                tcoeffs_std = tree.tree_map(np.zeros_like, tcoeffs_mean)
            else:

                def eps_like(s):
                    return inexact_eps * np.ones_like(s)

                tcoeffs_std = tree.tree_map(eps_like, tcoeffs_mean)
        else:

            def std_init(s: Array) -> Array:
                if s.dtype != np.dtype(bool):
                    msg = "Boolean entries expected in `is_exact`."
                    msg += f" Received: dtype={np.dtype(s)}"
                    raise TypeError(msg)
                return np.where(s, 0.0, inexact_eps)

            tcoeffs_std = tree.tree_map(std_init, is_exact)

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: np.shape(a) == np.shape(b), A, B)

        if not tree.tree_all(shape_equal(tcoeffs_mean, tcoeffs_std)):
            msg = "Input 'is_exact' has the wrong PyTree structure."
            msg += f" Expected: {tree.tree_map(np.shape, tcoeffs_mean)}."
            msg += f" Received: {tree.tree_map(np.shape, is_exact)}."
            raise ValueError(msg)
        return tcoeffs_std

    def _add_diffuse_derivatives(
        self, tcoeffs_mean, tcoeffs_std, /, *, diffuse_derivatives, diffuse_eps
    ):
        # Always set the mean to zero (for now at least).
        zeros = tree.tree_map(np.zeros_like, tcoeffs_mean[0])
        tcoeffs_mean = [*tcoeffs_mean, *[zeros for _ in range(diffuse_derivatives)]]

        unknowns = tree.tree_map(
            lambda s: diffuse_eps * np.ones_like(s), tcoeffs_std[0]
        )
        tcoeffs_std = [*tcoeffs_std, *[unknowns for _ in range(diffuse_derivatives)]]
        return tcoeffs_mean, tcoeffs_std

    def _process_base_scale(self, base_scale, single_flat, single_unravel):
        # Process the expected shape of the base-scale
        base_scale_expected = single_unravel(np.ones_like(single_flat))

        # If no base-scale is provided, use the default
        if base_scale is None:
            base_scale, _ = tree.ravel_pytree(base_scale_expected)
            return linalg.diagonal_matrix(base_scale)

        # Otherwise, check the shape and turn the scale into a matrix
        base_scale = tree.tree_map(np.asarray, base_scale)

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: np.shape(a) == np.shape(b), A, B)

        if tree.tree_structure(base_scale) != tree.tree_structure(base_scale_expected):
            msg = "The 'base_scale' argument has an unexpected PyTree structure."
            msg += f" Expected: {tree.tree_structure(base_scale_expected)}."
            msg += f" Received: {tree.tree_structure(base_scale)}."
            raise TypeError(msg)

        if not tree.tree_all(shape_equal(base_scale, base_scale_expected)):
            msg = "The base-scale has the wrong shape."
            msg += f" Expected: {tree.tree_map(np.shape, base_scale_expected)}."
            msg += f" Received: {base_scale.shape}."
            raise ValueError(msg)

        # Flatten the scale into something compatible with the flattened SSM
        base_scale, _ = tree.ravel_pytree(base_scale)
        return linalg.diagonal_matrix(base_scale)

    def constraint_ode_ts0(self, ode: problems.JetOde, /) -> DenseOdeTs0:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return DenseOdeTs0(ode=ode)

    def constraint_ode_ts1(self, ode: problems.JetOde, /) -> DenseOdeTs1:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return DenseOdeTs1(ode=ode)

    def constraint_ode_ts1_projected(
        self, ode: problems.JetOde, /, key, num_probes
    ) -> DenseOdeTs1Projected:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return DenseOdeTs1Projected(ode=ode, key=key, num_probes=num_probes)

    def constraint_residual(
        self,
        residual: problems.JetResidual,
        *,
        taylor_point: taylor_points.TaylorPoint | None = None,
    ) -> DenseResidual:
        if not isinstance(residual, problems.JetResidual):
            raise TypeError(residual)
        if taylor_point is None:
            taylor_point = taylor_points.taylor_point_prior()
        return DenseResidual(residual, taylor_point=taylor_point)
