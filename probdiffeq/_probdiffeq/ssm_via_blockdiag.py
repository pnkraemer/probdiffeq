from probdiffeq._probdiffeq import ssm_via_api as interfaces
from probdiffeq._probdiffeq import utilities
from probdiffeq._probdiffeq.problem_types import ODEFunction
from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Sequence, TypeVar
from probdiffeq.util import cholesky_util

__all__ = ["state_space_model_blockdiag"]


C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""


class BlockDiagLatentCond(interfaces.AbstractLatentCond):
    """Block-diagonal implementation of LatentCond operations."""

    def apply_flat(self, x, /):
        def apply_unbatch(s, c):
            s = c.to_latent * s
            m_new = c.to_observed * (c.A @ s + c.noise.mean_flat)
            c_new = c.to_observed[:, None] * c.noise.cholesky_flat
            return BlockDiagNormal(m_new, c_new, self.noise.tree_flatten)

        return func.vmap(apply_unbatch)(x, self)

    def marginalise(self, rv, /):
        matrix, noise = self.A, self.noise
        assert matrix.ndim == 3
        mean = self.to_latent * rv.mean_flat
        cholesky = self.to_latent[:, :, None] * rv.cholesky_flat

        mean_marg = np.einsum("ijk,ik->ij", matrix, mean) + noise.mean_flat

        chol1 = _transpose(matrix @ cholesky)
        chol2 = _transpose(noise.cholesky_flat)
        R_stack = (chol1, chol2)
        cholesky = func.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack)

        mean_new = self.to_observed * mean_marg
        cholesky_new = self.to_observed[:, :, None] * _transpose(cholesky)
        return BlockDiagNormal(mean_new, cholesky_new, self.noise.tree_flatten)

    def merge(self, other: "BlockDiagLatentCond", /) -> "BlockDiagLatentCond":
        # self = cond1 (outer), other = cond2 (inner)
        T = self.to_latent * other.to_observed

        A1, A2 = self.A, T[:, :, None] * other.A
        g = func.vmap(lambda a, b: a @ b)(A1, A2)

        m1, m2 = T * other.noise.mean_flat, self.noise.mean_flat
        xi = func.vmap(lambda a, b, c: a @ b + c)(A1, m1, m2)

        C1, C2 = self.noise.cholesky_flat, T[:, :, None] * other.noise.cholesky_flat
        R1 = _transpose(func.vmap(lambda a, b: a @ b)(A1, C2))
        R2 = _transpose(C1)
        Xi = func.vmap(cholesky_util.sum_of_sqrtm_factors)((R1, R2))
        Xi = _transpose(Xi)

        noise = BlockDiagNormal(xi, Xi, self.noise.tree_flatten)
        return BlockDiagLatentCond(
            g, noise, to_latent=other.to_latent, to_observed=self.to_observed
        )

    def revert(self, rv, /, *, solve_triu):
        mean = self.to_latent * rv.mean_flat
        cholesky = self.to_latent[:, :, None] * rv.cholesky_flat

        rv_chol_upper = _transpose(cholesky)
        noise_chol_upper = _transpose(self.noise.cholesky_flat)
        A_rv_chol_upper = _transpose(self.A @ cholesky)

        revert_conditional = func.partial(
            cholesky_util.revert_conditional, solve_triu=solve_triu
        )
        revert_vmap = func.vmap(revert_conditional)
        r_obs, (r_cor, gain) = revert_vmap(
            A_rv_chol_upper, rv_chol_upper, noise_chol_upper
        )
        cholesky_obs = np.transpose(r_obs, axes=(0, 2, 1))
        cholesky_cor = np.transpose(r_cor, axes=(0, 2, 1))

        mean_observed = (self.A @ mean[..., None])[..., 0] + self.noise.mean_flat
        mean_corrected = mean - (gain @ (mean_observed[..., None]))[..., 0]
        corrected = BlockDiagNormal(mean_corrected, cholesky_cor, rv.tree_flatten)
        bwd = BlockDiagLatentCond(
            gain,
            corrected,
            to_latent=1 / self.to_observed,
            to_observed=1 / self.to_latent,
        )

        mean_observed = self.to_observed * mean_observed
        cholesky_observed = self.to_observed[:, :, None] * cholesky_obs
        observed = BlockDiagNormal(
            mean_observed, cholesky_observed, self.noise.tree_flatten
        )
        return observed, bwd

    def preconditioner_apply(self, /):
        A = self.to_observed[:, :, None] * self.A * self.to_latent[:, None, :]
        mean = self.to_observed * self.noise.mean_flat
        cholesky = self.to_observed[:, :, None] * self.noise.cholesky_flat
        noise = BlockDiagNormal(mean, cholesky, self.noise.tree_flatten)
        to_observed = np.ones_like(self.to_observed)
        to_latent = np.ones_like(self.to_latent)
        return BlockDiagLatentCond(
            A, noise, to_observed=to_observed, to_latent=to_latent
        )


BlockDiagLatentCond._register_as_pytree()


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))


class state_space_model_blockdiag(interfaces.StateSpaceModel):
    """Implementation of block-diagonal SSM constructors."""

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
        init = BlockDiagNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        single_flat, single_unravel = tree.ravel_pytree(tcoeffs_mean[0])
        coeff_like = np.ones_like(single_flat)
        base_scale_expected = single_unravel(coeff_like)

        if output_scale is None:
            output_scale = base_scale_expected
        else:
            if tree.tree_structure(output_scale) != tree.tree_structure(
                base_scale_expected
            ):
                msg = "The 'output_scale' argument has an unexpected PyTree structure."
                msg += f" Expected: {tree.tree_structure(base_scale_expected)}."
                msg += f" Received: {tree.tree_structure(output_scale)}."
                raise TypeError(msg)

            output_scale = tree.tree_map(np.asarray, output_scale)

            def shape_equal(A, B):
                return tree.tree_map(lambda a, b: np.shape(a) == np.shape(b), A, B)

            if not tree.tree_all(shape_equal(output_scale, base_scale_expected)):
                msg = "The output-scale has the wrong shape."
                msg += f" Expected: {tree.tree_map(np.shape, base_scale_expected)}."
                msg += f" Received: {output_scale.shape}."
                raise ValueError(msg)

        output_scale, _ = tree.ravel_pytree(output_scale)

        num_derivatives = len(tcoeffs_mean) - 1
        a, q_sqrtm = utilities.system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1,))
        precon_fun = utilities.preconditioner_taylor(num_derivatives=num_derivatives)

        _base_scale = output_scale

        def discretise(dt, output_scale: Array | None = None):
            p, p_inv = precon_fun(dt)
            if output_scale is None:
                output_scale = np.ones_like(_base_scale)
            else:
                output_scale = np.asarray(output_scale)

                if output_scale.shape != _base_scale.shape:
                    msg = "The output-scale has the wrong shape."
                    msg += f" Expected: {_base_scale.shape}."
                    msg += f" Received: {output_scale.shape}."
                    raise ValueError(msg)
                output_scale, _ = tree.ravel_pytree(output_scale)

            scale = output_scale * _base_scale
            # Flatten the scale into something compatible with the flattened SSM
            (d,) = scale.shape

            A_batch = np.ones((d, 1, 1)) * a[None, :, :]
            mean = np.ones((d, 1)) * q0[None, :]
            cholesky = scale[:, None, None] * q_sqrtm[None, :, :]
            tree_flatten = BlockDiagTreeFlatten.from_example(tcoeffs_mean)
            noise = BlockDiagNormal(mean, cholesky, tree_flatten)
            p = np.ones((d, 1)) * p[None, :]
            p_inv = np.ones((d, 1)) * p_inv[None, :]
            return BlockDiagLatentCond(A_batch, noise, to_latent=p_inv, to_observed=p)

        return init, discretise

    def prior_exponential(
        self,
        ode,
        tcoeffs_mean: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        del ode, tcoeffs_mean, is_exact, inexact_eps, diffuse_derivatives
        del diffuse_eps, output_scale
        msg = "Block-diagonal exponential priors have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def prior_exponential_diffuse(
        self,
        ode,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        del ode, tcoeffs_mean, tcoeffs_std, diffuse_derivatives
        del diffuse_eps, output_scale
        msg = "Block-diagonal exponential priors have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def constraint_ode_ts0(self, ode, /) -> "BlockDiagOdeTs0":
        if not isinstance(ode, ODEFunction):
            raise TypeError(ode)
        return BlockDiagOdeTs0(ode=ode)

    def constraint_ode_ts1(self, ode, /) -> "BlockDiagOdeTs1":
        if not isinstance(ode, ODEFunction):
            raise TypeError(ode)
        if ode.num_derivatives_in_args > 2:
            msg = "This linearization is not compatible with high-order ODEs as of yet."
            raise ValueError(msg)
        return BlockDiagOdeTs1(ode=ode)

    def constraint_residual(self, residual, *, linearization_point=None):
        raise NotImplementedError

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


class BlockDiagOdeTs0(interfaces.AbstractOde):
    """Construct a block-diagonal implementation of ODE-TS0 linearization."""

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv, state: None, *, damp: float, t):
        del state

        jet_coords = rv.mean[: self.ode.num_derivatives_in_args]
        fx = self.ode.vector_field(jet_coords=jet_coords, t=t)
        fx = tree.tree_map(lambda s: -s, fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)

        def a1(s):
            return s[[self.ode.num_derivatives_in_args], ...]

        linop = func.vmap(func.jacrev(a1))(rv.mean_flat)

        cond = BlockDiagLatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class BlockDiagOdeTs1(interfaces.AbstractOde):
    """Construct a block-diagonal implementation of ODE-TS1 linearization."""

    def init_linearization(self):
        return self.ode.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t):

        m0_tree = rv.mean[0]
        rv0 = BlockDiagNormal.from_dirac([m0_tree], damp=0.0)

        def a1(s):
            return s[[1], ...]

        linop = func.vmap(func.jacrev(a1))(rv.mean_flat)

        def vf_flat(u):
            u_tree = rv0.tree_flatten.unflatten_array(u[:, None])
            fu_tree = self.ode.vector_field(jet_coords=u_tree, t=t)
            return rv0.tree_flatten.flatten_tree([fu_tree]).reshape((-1,))

        # Evaluate the linearisation
        m0 = rv.mean_flat[:, 0]
        fx, J_diag, state = self.ode.jacobian.calculate_diagonal(vf_flat, m0, state)

        E1 = func.jacrev(lambda s: s[0])(rv.mean_flat[0])
        linop = linop - J_diag[:, None, None] * E1[None, None, :]

        fx = rv.mean_flat[:, 1] - fx
        fx = fx[..., None]
        diff = func.vmap(lambda a, b: a @ b)(linop, rv.mean_flat)
        fx = fx - diff
        fx = rv0.tree_flatten.unflatten_array(fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)
        cond = BlockDiagLatentCond.from_linop_and_noise(linop, bias)
        return cond, state


@structs.dataclass
class BlockDiagTreeFlatten(interfaces.AbstractTreeFlatten):
    """Flattening information for block-diagonal random variables."""

    # The treedef of the target
    treedef: Any

    # A map that unravels each leaf. Note that this is not the same
    # as unravelling each Taylor coefficient, because the Taylor coefficients
    # themselves can be pytrees, whereas the leaves are always arrays.
    # The below function exclusively reshapes arrays
    unravel_leaf: Any

    def flatten_tree(self, x):
        leaves = tree.tree_leaves_depth_one(x)
        leaves_flat = [tree.ravel_pytree(s)[0] for s in leaves]
        return np.stack(leaves_flat).T

    def unflatten_array(self, x):
        x_tree = tree.tree_unflatten(self.treedef, [*x.T])
        return tree.tree_map(self.unravel_leaf, x_tree)

    @classmethod
    def from_example(cls, x):
        leaves, treedef = tree.tree_flatten_depth_one(x)
        _, unravel_leaf = tree.ravel_pytree(leaves[0])
        return cls(treedef, unravel_leaf)


class BlockDiagNormal(interfaces.AbstractTreeNormal[BlockDiagTreeFlatten]):
    """Construct a block-diagonal normal distribution.

    This assumes that the pytree is of the form [M_1, ..., M_{num_coeffs}],
    where each M_i is a pytree of the same structure.

    Shapes:
    - mean: (d, num_coeffs)
    - cholesky: (d, num_coeffs, num_coeffs)

    where d is the number of elements in each M.

    """

    @classmethod
    def from_dirac(cls, mean, *, damp):
        utilities.verify_taylor_coefficient_pytree(mean)

        std = tree.tree_map(lambda s: damp * np.ones_like(s), mean)
        return BlockDiagNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        utilities.verify_taylor_coefficient_pytree(mean)
        utilities.verify_taylor_coefficient_pytree(std)

        tree_flatten = BlockDiagTreeFlatten.from_example(mean)

        loc_flat = tree_flatten.flatten_tree(mean)
        scale_flat = tree_flatten.flatten_tree(std)
        num_coeffs = len(mean)

        # Promote std into covariance matrix and apply damping
        num_coeffs = len(mean)
        d = np.ones((num_coeffs,))

        cholesky = linalg.diagonal_matrix(d)
        cholesky_flat = scale_flat[..., None] * cholesky[None, ...]
        return cls(loc_flat, cholesky_flat, tree_flatten)

    @property
    def mean(self):
        return self._mean_batched()

    def _mean_batched(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(BlockDiagNormal._mean_batched)(self)

        return self.tree_flatten.unflatten_array(self.mean_flat)

    @property
    def std(self):
        return self._std_batched()

    def _std_batched(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(BlockDiagNormal._std_batched)(self)

        std_flat = func.vmap(func.vmap(linalg.vector_norm))(self.cholesky_flat)
        return self.tree_flatten.unflatten_array(std_flat)

    def residual_white_rms_tree(self, u, /):
        # todo: add sth like an "axis" argument to make it more obvious
        # to a user that this one here is vector-valued?

        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.residual_white_rms_flat(u_latent)

    def residual_white_rms_flat(self, u_latent, /):
        mean = np.reshape(self.mean_flat - u_latent, (-1,))
        cholesky = np.reshape(self.cholesky_flat, (-1,))
        return mean / cholesky / np.sqrt(mean.size)

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky_flat
        return BlockDiagNormal(self.mean_flat, cholesky, self.tree_flatten)

    def logpdf_tree(self, u, /):
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.logpdf_flat(u_latent)

    def logpdf_flat(self, u_latent, /):
        return np.sum(func.vmap(BlockDiagNormal.logpdf_scalar_flat)(self, u_latent))

    def logpdf_scalar_flat(self, u):
        cholesky = linalg.qr_r(self.cholesky_flat.T).T

        dx = u - self.mean_flat
        w = linalg.solve_triu(cholesky.T, dx, trans="T")

        maha_term = linalg.vector_dot(w, w)

        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * np.log(np.pi() * 2))

    def to_multivariate_normal(self):
        mean = np.reshape(self.mean_flat.T, (-1,), order="F")
        cov = np.block_diag(self._cov_dense())
        return mean, cov

    def _cov_dense(self):
        if self.cholesky_flat.ndim > 2:
            return func.vmap(BlockDiagNormal._cov_dense)(self)
        return self.cholesky_flat @ self.cholesky_flat.T

    def sample_tree(self, key):
        if self.cholesky_flat.ndim > 3:
            d, *_ = self.cholesky_flat.shape
            keys = random.split(key, num=d)
            return func.vmap(BlockDiagNormal.sample_tree)(self, keys)

        sample_latent = self.sample_flat(key)
        return self.tree_flatten.unflatten_array(sample_latent)

    def sample_flat(self, key):
        if self.cholesky_flat.ndim > 3:
            d, *_ = self.cholesky_flat.shape
            keys = random.split(key, num=d)
            return func.vmap(BlockDiagNormal.sample_flat)(self, keys)

        d, _n, n = self.cholesky_flat.shape
        base = random.normal(key, shape=(d, n))
        return self.mean_flat + np.einsum("ijk,ij->ik", self.cholesky_flat, base)

    def identity_conditional(self) -> BlockDiagLatentCond:
        (d, ndim) = self.mean_flat.shape
        m0 = np.zeros((d, ndim))
        c0 = np.zeros((d, ndim, ndim))
        noise = BlockDiagNormal(m0, c0, self.tree_flatten)
        matrix = np.ones((d, 1, 1)) * np.eye(ndim, ndim)[None, ...]
        return BlockDiagLatentCond.from_linop_and_noise(matrix, noise)

    def prototype_output_scale_calibrated(self):
        single_flat, _ = tree.ravel_pytree(self.mean[0])
        # TODO: technically, these should be pytrees according
        # to the leaf structure, right?
        return np.ones(single_flat.shape)

    def to_derivative(self, i, std):
        def select(a):
            return np.asarray([a[i]])

        (d, n) = self.mean_flat.shape
        x = np.zeros((d, n))
        linop = func.vmap(func.jacrev(select))(x)

        u_like = tree.tree_map(np.zeros_like, self.mean[0])
        noise = BlockDiagNormal.from_mean_and_std([u_like], [std])
        return BlockDiagLatentCond.from_linop_and_noise(linop, noise)

    @staticmethod
    def register_pytree_node() -> None:
        def flatten(normal):
            children = normal.mean_flat, normal.cholesky_flat
            aux = (normal.tree_flatten,)
            return children, aux

        def unflatten(aux, children):
            (tree_flatten,) = aux
            mean, cholesky = children
            return BlockDiagNormal(mean, cholesky, tree_flatten)

        tree.register_pytree_node(BlockDiagNormal, flatten, unflatten)


BlockDiagNormal.register_pytree_node()
