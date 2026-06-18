from probdiffeq._probdiffeq import problems, ssm_impl_api, taylor_points, utilities
from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Callable, Sequence, TypeVar
from probdiffeq.util import cholesky_util

__all__ = ["state_space_model_blockdiag"]


C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""


class BlockDiagMatrix(ssm_impl_api.AbstractLinOp):
    def __init__(self, *, matrix_dnm):
        d, n_out, n_in = matrix_dnm.shape
        super().__init__(n_in=n_in, n_out=n_out, d_in=d, d_out=d)
        self.matrix_dnm = matrix_dnm

    def matvec_flat(self, vec, /):
        raise NotImplementedError

    def matvec_ndmd(self, vec, /):
        raise NotImplementedError

    def matvec_dndm(self, vec, /):
        return linalg.einsum("...nm,...m->...n", self.matrix_dnm, vec)

    @property
    def precon_prototype(self):
        return np.ones((self.d_in,)), np.ones((self.d_out,))


class MatfreeLinOpODEConstraint(ssm_impl_api.AbstractLinOp):
    def __init__(self, *, ode, tree_flatten, x, t):
        self.ode = ode
        self.tree_flatten = tree_flatten
        self.t = t
        self.x = x

        *_batch, n_in, d_in = x.shape
        x_like = structs.ShapeDtypeStruct(shape=(n_in, d_in), dtype=x.dtype)
        n_out, d_out = func.eval_shape(self._information_operator, x_like).shape
        super().__init__(n_in=n_in, n_out=n_out, d_in=d_in, d_out=d_out)

    def matvec_ndmd(self, vec):
        # TODO: this line implies we do sooo many unnecessary function evals...
        _, jvp = func.linearize(self._information_operator, self.x)

        x1 = vec[np.asarray([self.ode.num_tcoeffs_in_args])]
        fx0 = jvp(vec)
        return x1 - fx0

    def _information_operator(self, s: Array) -> Array:
        # Move to latent space (arg is (n, d) but latent space is (d, n))
        s = s.T

        # Extract all tcoeffs
        jet_coords = self.tree_flatten.unflatten_array(s)

        # Extract relevant tcoeffs ("jet-coordinates")
        jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

        # Evaluate the actual vector field
        fs0 = self.ode.vector_field(jet_coords=jet_coords, t=self.t)

        # Bring back into (m, d) form.
        return tree.ravel_pytree(fs0)[0][None, :]

    def matvec_dndm(self, vec_dn):
        vec_nd = vec_dn.T
        return self.matvec_ndmd(vec_nd).T

    def matvec_flat(self, vec_flat):
        vec = vec_flat.reshape((self.n_in, self.d_in))
        return self.matvec_ndmd(vec).reshape((-1,))

    @property
    def precon_prototype(self):
        return np.ones((self.d_in,)), np.ones((self.d_out,))

    @classmethod
    def _register_as_pytree(cls) -> None:
        """Register this class (or a subclass) as a JAX pytree."""

        def flatten(linop):
            children = linop.t, linop.x
            aux = (linop.ode, linop.tree_flatten)
            return children, aux

        def unflatten(aux, children):
            (t, x) = children
            ode, tree_flatten = aux
            return cls(ode=ode, tree_flatten=tree_flatten, t=t, x=x)

        tree.register_pytree_node(cls, flatten, unflatten)


MatfreeLinOpODEConstraint._register_as_pytree()


class MatfreeLinOpLstSq(ssm_impl_api.AbstractLinOp):
    def __init__(self, cholesky_x, cholesky_yx, matfree_linop):
        super().__init__(
            n_in=matfree_linop.n_out,
            n_out=matfree_linop.n_in,
            d_in=matfree_linop.d_out,
            d_out=matfree_linop.d_in,
        )
        self.cholesky_x = cholesky_x
        self.cholesky_yx = cholesky_yx
        self.matfree_linop = matfree_linop

    @property
    def precon_prototype(self):
        return np.ones((self.d_in,)), np.ones((self.d_out,))

    def matvec_dndm(self, vec_dn):
        vec_flat = vec_dn.T.reshape((self.n_in * self.d_in,))
        Avec_flat = self.matvec_flat(vec_flat)
        return Avec_flat.reshape((self.n_out, self.d_out)).T

    def matvec_ndmd(self, vec):
        vec_flat = vec.reshape((self.n_in * self.d_in,))
        Avec_flat = self.matvec_flat(vec_flat)
        return Avec_flat.reshape((self.n_out, self.d_out))

    def matvec_flat(self, vec):
        # LSMR struggles with zero RHS's right now,
        # so we shortcut.
        out_like = np.zeros((self.n_out * self.d_out,))
        cond = linalg.vector_norm(vec) == 0
        return np.where(cond, out_like, self._matvec_flat_nonzero(vec))

    def _matvec_flat_nonzero(self, vec):

        def vecmat(s):
            # vector-Jacobian product
            Ats = self.matfree_linop.vecmat_flat(s)

            # Cholesky-vector products
            BtAts = self.matvec_bd_transpose(self.cholesky_x, Ats)
            Dts = self.matvec_bd_transpose(self.cholesky_yx, s)
            return np.concatenate([BtAts, Dts], axis=0)

        # TODO: turn "D" into a damping factor
        tol = np.finfo_eps(vec.dtype)
        lstsq_sol = linalg.lstsq_lsmr(vecmat, vec, atol=tol, btol=tol, ctol=tol)
        return self.matvec_bd(self.cholesky_x, lstsq_sol[: (self.n_out * self.d_out)])

    @staticmethod
    def matvec_bd_transpose(M, v):
        d, n, _m = M.shape
        v_dn = v.reshape((n, d)).T
        # transpose matvec, not matvec:
        Mv = linalg.einsum("...mn,...m->...n", M, v_dn)
        return Mv.T.reshape((-1,))

    @staticmethod
    def matvec_bd(M, v):
        d, n, _m = M.shape
        v_dn = v.reshape((n, d)).T
        Mv = linalg.einsum("...nm,...m->...n", M, v_dn)
        return Mv.T.reshape((-1,))


class BlockDiagLatentCond(ssm_impl_api.AbstractLatentCond):
    """Block-diagonal implementation of LatentCond operations."""

    def apply_flat(self, x, /):
        x_latent = self.to_latent * x
        m_new = self.A.matvec_dndm(x_latent) + self.noise.mean_flat
        m_obs = self.to_observed * m_new
        c_obs = self.to_observed[..., None] * self.noise.cholesky_flat
        return BlockDiagNormal(m_obs, c_obs, self.noise.tree_flatten)

    def marginalise(self, rv, /):
        matrix, noise = self.A, self.noise
        mean = self.to_latent * rv.mean_flat
        cholesky = self.to_latent[:, :, None] * rv.cholesky_flat

        mean_marg = matrix.matvec_dndm(mean) + noise.mean_flat

        chol1 = _transpose(matrix.matmat_dndm(cholesky))
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


class BlockDiagLatentCondProjected(ssm_impl_api.AbstractLatentCond):
    def apply_flat(self, x, /):
        y = self.A.matvec_ndmd(x.T).T
        m = y + self.noise.mean_flat
        c = self.noise.cholesky_flat
        tree_flatten = self.noise.tree_flatten
        return BlockDiagNormal(m, c, tree_flatten)

    def marginalise(self, rv, /):

        # Observed mean
        obs_mean = self.A.matvec_dndm(rv.mean_flat) + self.noise.mean_flat

        # TODO: use precon
        # TODO: we currently ignore the noise?
        key = random.prng_key(seed=1)  # TODO: this seed should be in some form of state
        d, n = rv.mean_flat.shape
        S = 2 * n
        normals = random.rademacher(key, shape=(S, n, d), dtype=rv.mean_flat.dtype)
        chols = func.vmap(self.A.matvec_ndmd)(normals)
        chols /= np.sqrt(S)
        obs_chol = func.vmap(lambda s: linalg.qr_r(s).T, in_axes=-1)(chols)
        return BlockDiagNormal(obs_mean, obs_chol, self.noise.tree_flatten)

        # raise RuntimeError
        # # Sample count. See p. 6 in
        # # https://arxiv.org/abs/2606.08203
        # num = len(rv.mean) * 2

        # # Samples
        # key = random.prng_key(seed=1)
        # key_rv, key_noise = random.split(key, num=2)
        # keys = random.split(key_rv, num=num)
        # X = func.vmap(rv.sample_flat)(keys)
        # keys = random.split(key_noise, num=num)
        # Y = func.vmap(self.noise.sample_flat)(keys)

        # # Observed covariance
        # AX = func.vmap(lambda s: self.A.matvec_ndmd(s.T).T)(X) + Y
        # AX -= obs_mean[None, ...]
        # AX /= np.sqrt(AX.shape[0] - 1)
        # obs_chol = func.vmap(lambda s: linalg.qr_r(s).T, in_axes=1)(AX)
        # return BlockDiagNormal(obs_mean, obs_chol, self.noise.tree_flatten)

    def revert(self, rv, /, *, solve_triu: Callable):
        del solve_triu  # unused
        # TODO: use preconditioning (currently assume P=1)

        # Sample count. See p. 6 in
        # https://arxiv.org/abs/2606.08203
        num = len(rv.mean) * 2

        # Observed mean
        obs_mean = self.A.matvec_dndm(rv.mean_flat) + self.noise.mean_flat

        # Gain & conditioning
        K = MatfreeLinOpLstSq(rv.cholesky_flat, self.noise.cholesky_flat, self.A)

        # Posterior mean:
        cond_mean = rv.mean_flat - K.matvec_dndm(
            self.A.matvec_dndm(rv.mean_flat) + self.noise.mean_flat
        )
        # Samples
        key = random.prng_key(seed=1)  # TODO: this seed should be in some form of state
        n = len(rv.mean)
        d = rv.mean_flat.size // n
        S = 2 * n
        normals = random.rademacher(key, shape=(S, n, d), dtype=rv.mean_flat.dtype)
        normals = linalg.einsum("dnm,smd->snd", rv.cholesky_flat, normals)

        # Posteriors
        chols = func.vmap(lambda s: s - K.matvec_ndmd(self.A.matvec_ndmd(s)))(normals)
        chols /= np.sqrt(S)
        cond_chol = func.vmap(lambda s: linalg.qr_r(s).T, in_axes=-1)(chols)
        noise = BlockDiagNormal(cond_mean, cond_chol, rv.tree_flatten)
        cond = BlockDiagLatentCondProjected.from_linop_and_noise(K, noise)

        # Marginals (redo samples)
        S = 2 * n
        normals = random.rademacher(key, shape=(S, n, d), dtype=rv.mean_flat.dtype)
        chols = func.vmap(self.A.matvec_ndmd)(normals)
        chols /= np.sqrt(S)
        obs_chol = func.vmap(lambda s: linalg.qr_r(s).T, in_axes=-1)(chols)
        observed = BlockDiagNormal(obs_mean, obs_chol, self.noise.tree_flatten)

        # Group and return
        return observed, cond

    def merge(self, other, /):
        raise NotImplementedError

    def preconditioner_apply(self, /):
        raise NotImplementedError


BlockDiagLatentCondProjected._register_as_pytree()


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))


class BlockDiagOdeTs0(ssm_impl_api.AbstractOde):
    """Block-diagonal ODE linearization via TS0 (zeroth-degree Taylor series: evaluate at the or mean, no Jacobian)."""

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv, state: None, *, damp: float, t):
        del state

        jet_coords = rv.mean[: self.ode.num_tcoeffs_in_args]
        fx = self.ode.vector_field(jet_coords=jet_coords, t=t)
        fx = tree.tree_map(lambda s: -s, fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)

        def a1(s):
            return s[[self.ode.num_tcoeffs_in_args], ...]

        linop = func.vmap(func.jacrev(a1))(rv.mean_flat)

        cond = BlockDiagLatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class BlockDiagOdeTs1(ssm_impl_api.AbstractOde):
    """Block-diagonal ODE linearization via TS1 (first-degree Taylor series: evaluate the residual and its Jacobian at the linearization point)."""

    def init_linearization(self):
        return self.ode.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t):

        # Understand flattening and unflattening for
        # single Taylor coefficients
        rv0 = BlockDiagNormal.from_dirac([rv.mean[0]], damp=damp)

        def a1(s):
            return s[[self.ode.num_tcoeffs_in_args], ...]

        linop = func.vmap(func.jacrev(a1))(rv.mean_flat)

        def vf_flat(u):
            u_tree = rv.tree_flatten.unflatten_array(u.T)
            u_tree = u_tree[: self.ode.num_tcoeffs_in_args]
            fu_tree = self.ode.vector_field(jet_coords=u_tree, t=t)
            return rv0.tree_flatten.flatten_tree([fu_tree]).T

        # Evaluate the linearisation
        # Not 100% the most efficient because we compute the diagonal of
        # the function of all tcoeffs instead of just the relevant ones.
        # But if we use reverse-Jacobians, things should be fine.
        fx, J_diag, state = self.ode.jacobian.calculate_diagonal_along_d(
            vf_flat, rv.mean_flat.T, state
        )

        # Jacobian objects work with (n, d) arrays but block-diagonal models with (d, n) arrays
        fx = fx.T

        # J_diag.shape = (d, 1, n)
        # linop.shape: (d, 1, n)
        linop = linop - J_diag
        fx = rv.mean_flat[:, self.ode.num_tcoeffs_in_args][:, None] - fx
        diff = func.vmap(lambda a, b: a @ b)(linop, rv.mean_flat)
        fx = fx - diff
        fx = rv0.tree_flatten.unflatten_array(fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)
        cond = BlockDiagLatentCond.from_linop_and_noise(linop, bias)
        return cond, state


class BlockDiagOdeTs1Projected(ssm_impl_api.AbstractOde):
    """Dense ODE linearization via TS1 (first-degree Taylor series: evaluate the residual and its Jacobian at the linearization point)."""

    def init_linearization(self):
        return self.ode.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t: float):
        # Read n and d so that we can turn latent arrays into
        # (n, d) arrays, which Jacobians require
        m_tree = rv.mean
        n = len(m_tree)
        d = rv.mean_flat.size // n

        # Materialize the Jacobian
        m0 = rv.mean_flat.T
        fun = func.partial(self.information_operator, tree_flatten=rv.tree_flatten, t=t)
        fx, JVP = func.linearize(fun, m0)

        # Flatten fx and J correctly (from [m, d, n, d] to [md,nd])
        m, d = fx.shape
        fx = fx.reshape((m * d,))

        # Complete the expressions for bias and linop

        fx = JVP(rv.mean_flat.T) - fx
        # E1 = projection_e1(rv.mean_flat.T)

        # Flatten fx into the correct pytree structure
        f0 = BlockDiagNormal.from_dirac(
            [m_tree[self.ode.num_tcoeffs_in_args]], damp=damp
        )
        fx = f0.tree_flatten.unflatten_array(fx.T)

        # Collect all quantities and return
        noise = BlockDiagNormal.from_dirac(fx, damp=damp)

        # n_out, d_out, n_in, d_in = E1.shape
        linop = MatfreeLinOpODEConstraint(
            ode=self.ode, tree_flatten=rv.tree_flatten, t=t, x=m0
        )
        cond = BlockDiagLatentCondProjected.from_linop_and_noise(linop, noise)
        return cond, state

    # Rewrite the vector field as one that maps
    # Arrays to arrays.
    # Maps (n, d) to (1, d) to conform the Jacobian API
    def information_operator(self, s: Array, *, tree_flatten, t) -> Array:
        # Move to latent space (arg is (n, d) but latent space is (d, n))
        s = s.T

        # Extract all tcoeffs
        jet_coords = tree_flatten.unflatten_array(s)

        # Extract relevant tcoeffs ("jet-coordinates")
        jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

        # Evaluate the actual vector field
        fs0 = self.ode.vector_field(jet_coords=jet_coords, t=t)

        # Bring back into (m, d) form.
        return tree.ravel_pytree(fs0)[0][None, :]


@structs.dataclass
class BlockDiagTreeFlatten(ssm_impl_api.AbstractTreeFlatten):
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


class BlockDiagNormal(ssm_impl_api.AbstractTreeNormal[BlockDiagTreeFlatten]):
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

    @classmethod
    def from_dense_via_truncation(cls, rv):
        n = len(rv.mean)
        d = rv.mean_flat.size // n
        cholesky = BlockDiagNormal._remove_offdiag(rv.cholesky_flat, n=n, d=d)
        mean = rv.mean_flat.reshape((n, d)).T

        tree_flatten = BlockDiagNormal.from_dirac(rv.mean, damp=0.0).tree_flatten
        return cls(mean, cholesky, tree_flatten)

    @staticmethod
    def _remove_offdiag(cholesky, n, d):
        cholesky_4tensor = cholesky.reshape((n, d, n, d))
        cholesky_removed = np.einsum("ijkl,jl->ijkl", cholesky_4tensor, np.eye(d))
        return np.einsum("ndmd->dnm", cholesky_removed)

    def to_dense_normal(self):
        # Transpose before flattening
        mean = self.mean_flat.T.reshape((-1,))
        d, n = self.mean_flat.shape

        cholesky = np.einsum("dnm,dt->ndmt", self.cholesky_flat, np.eye(d))
        cholesky = cholesky.reshape((d * n, d * n))

        from probdiffeq._probdiffeq import ssm_impl_dense

        _, unravel = tree.ravel_pytree(self.mean)
        tree_flatten = ssm_impl_dense.DenseTreeFlatten(unravel)
        return ssm_impl_dense.DenseNormal(mean, cholesky, tree_flatten)

    @property
    def batch_shape(self):
        *shape, _d, _n = self.mean_flat.shape
        return shape

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

    def residual_whitened_rms_tree(self, u, /):
        # todo: add sth like an "axis" argument to make it more obvious
        # to a user that this one here is vector-valued?

        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.residual_whitened_rms_flat(u_latent)

    def residual_whitened_rms_flat(self, u_latent, /):
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
        mean = np.reshape(self.mean_flat.T, (-1,))

        cov = self._cov_dense()
        *_batch, d, n, _n = cov.shape
        eye = np.eye(d)
        cov_full = linalg.einsum("dnm,dt->ndmt", cov, eye)
        cov = cov_full.reshape((n * d, -1))
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

        d, n, _n = self.cholesky_flat.shape
        base = random.normal(key, shape=(d, n))
        return self.mean_flat + np.einsum("ijk,ik->ij", self.cholesky_flat, base)

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


class BlockDiagWienerIntegrated(ssm_impl_api.AbstractPrior):
    def __init__(self, init, output_scale, *, a, q_sqrtm, q0, tree_flatten, precon_fun):
        super().__init__(init, output_scale)
        self.a = a
        self.q_sqrtm = q_sqrtm
        self.q0 = q0
        self.tree_flatten = tree_flatten
        self.precon_fun = precon_fun

    def transition(self, *, dt: float, output_scale: Array) -> BlockDiagLatentCond:
        p, p_inv = self.precon_fun(dt)

        output_scale = np.asarray(output_scale)
        if output_scale.shape != self.output_scale.shape:
            msg = "The output-scale has the wrong shape."
            msg += f" Expected: {output_scale.shape}."
            msg += f" Received: {self.output_scale.shape}."
            raise ValueError(msg)

        output_scale = self.output_scale * output_scale

        # Flatten the scale into something compatible with the flattened SSM
        (d,) = output_scale.shape

        mean = np.ones((d, 1)) * self.q0[None, :]
        cholesky = output_scale[:, None, None] * self.q_sqrtm[None, :, :]
        noise = BlockDiagNormal(mean, cholesky, self.tree_flatten)

        A_batch = np.ones((d, 1, 1)) * self.a[None, :, :]
        A = BlockDiagMatrix(matrix_dnm=A_batch)
        p = np.ones((d, 1)) * p[None, :]
        p_inv = np.ones((d, 1)) * p_inv[None, :]
        return BlockDiagLatentCond(A, noise, to_latent=p_inv, to_observed=p)

    @staticmethod
    def register_pytree():
        def flatten(iwp):
            children = (iwp.init, iwp.output_scale, iwp.a, iwp.q_sqrtm, iwp.q0)
            aux = (iwp.tree_flatten, iwp.precon_fun)
            return children, aux

        def unflatten(aux, children):
            tf, precon_fun = aux
            init, output_scale, a, q_sqrtm, q0 = children
            return BlockDiagWienerIntegrated(
                init,
                output_scale,
                a=a,
                q_sqrtm=q_sqrtm,
                q0=q0,
                tree_flatten=tf,
                precon_fun=precon_fun,
            )

        tree.register_pytree_node(BlockDiagWienerIntegrated, flatten, unflatten)


BlockDiagWienerIntegrated.register_pytree()


class state_space_model_blockdiag(ssm_impl_api.StateSpaceModel):
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
    ) -> BlockDiagWienerIntegrated:
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
    ) -> BlockDiagWienerIntegrated:
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
        output_scale_expected = single_unravel(coeff_like)

        if output_scale is None:
            output_scale = output_scale_expected
        else:
            expected_structure = tree.tree_structure(output_scale_expected)
            received_structure = tree.tree_structure(output_scale)
            if expected_structure != received_structure:
                msg = "The 'output_scale' argument has an unexpected PyTree structure."
                msg += f" Expected: {expected_structure}."
                msg += f" Received: {received_structure}."
                raise TypeError(msg)

            output_scale = tree.tree_map(np.asarray, output_scale)

            def shape_equal(A, B):
                return tree.tree_map(lambda a, b: np.shape(a) == np.shape(b), A, B)

            if not tree.tree_all(shape_equal(output_scale, output_scale_expected)):
                msg = "The output-scale has the wrong shape."
                msg += f" Expected: {tree.tree_map(np.shape, output_scale_expected)}."
                msg += f" Received: {output_scale.shape}."
                raise ValueError(msg)

        output_scale, _ = tree.ravel_pytree(output_scale)

        num_derivatives = len(tcoeffs_mean) - 1
        a, q_sqrtm = utilities.system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1,))
        tf = BlockDiagTreeFlatten.from_example(tcoeffs_mean)
        precon_fun = utilities.preconditioner_taylor(num_derivatives)
        return BlockDiagWienerIntegrated(
            init,
            output_scale,
            a=a,
            q_sqrtm=q_sqrtm,
            q0=q0,
            tree_flatten=tf,
            precon_fun=precon_fun,
        )

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

    def constraint_ode_ts0(self, ode: problems.JetOde, /) -> BlockDiagOdeTs0:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return BlockDiagOdeTs0(ode=ode)

    def constraint_ode_ts1(self, ode: problems.JetOde, /) -> BlockDiagOdeTs1:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return BlockDiagOdeTs1(ode=ode)

    def constraint_ode_ts1_projected(
        self, ode: problems.JetOde, /
    ) -> BlockDiagOdeTs1Projected:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)

        return BlockDiagOdeTs1Projected(ode=ode)

    def constraint_residual(
        self,
        residual: problems.JetResidual,
        *,
        taylor_point: taylor_points.TaylorPoint | None = None,
    ):
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
