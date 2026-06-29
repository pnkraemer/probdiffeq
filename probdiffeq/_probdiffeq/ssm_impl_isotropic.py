from probdiffeq._probdiffeq import problems, ssm_impl_api, taylor_points, utilities
from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Sequence, TypeVar
from probdiffeq.util import cholesky_util

T = TypeVar("T", bound=Array)
"""A type-variable for Array types.

For example, this variable is used for means and Cholesky factors
in normal distributions.
"""

C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""

__all__ = ["state_space_model_isotropic"]


@structs.dataclass
class IsotropicTreeFlatten(ssm_impl_api.AbstractTreeFlatten):
    """Flattening information for isotropic random variables."""

    # The treedef of the target
    treedef: Any

    # A map that unravels each leaf. Note that this is not the same
    # as unravelling each Taylor coefficient, because the Taylor coefficients
    # themselves can be pytrees, whereas the leaves are always arrays.
    # The below function exclusively reshapes arrays
    unravel_leaf: Any

    def flatten_tree(self, x):

        def is_leaf(s):
            return tree.tree_structure(s) == tree.tree_structure(x[0])

        leaves = tree.tree_leaves_depth_one(x)
        leaves_flat = [tree.ravel_pytree(s)[0] for s in leaves]
        return np.stack(leaves_flat)

    def flatten_tree_scalar(self, x):

        def is_leaf(s):
            return tree.tree_structure(s) == tree.tree_structure(x[0])

        leaves = tree.tree_leaves_depth_one(x)
        return np.stack(leaves)

    def unflatten_array(self, x):
        x_tree = tree.tree_unflatten(self.treedef, [*x])
        return tree.tree_map(self.unravel_leaf, x_tree)

    def unflatten_array_scalar(self, x):
        return tree.tree_unflatten(self.treedef, [*x])

    @classmethod
    def from_example(cls, x):
        def is_leaf(s):
            return tree.tree_structure(s) == tree.tree_structure(x[0])

        leaves, treedef = tree.tree_flatten_depth_one(x)
        _, unravel_leaf = tree.ravel_pytree(leaves[0])
        return cls(treedef, unravel_leaf)


class IsotropicLatentCond(ssm_impl_api.AbstractLatentCond):
    """Isotropic (scalar-variance) implementation of LatentCond operations."""

    def apply_flat(self, x, /):
        x = self.to_latent[:, None] * x
        mean_new = self.to_observed[:, None] * self.A @ x + self.noise.mean_flat
        cholesky_new = np.abs(self.to_observed[:, None]) * self.noise.cholesky_flat
        return IsotropicNormal(mean_new, cholesky_new, self.noise.tree_flatten)

    def marginalise(self, rv, /):
        mean = self.to_latent[:, None] * rv.mean_flat
        cholesky = self.to_latent[:, None] * rv.cholesky_flat
        R_stack = ((self.A @ cholesky).T, self.noise.cholesky_flat.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = self.to_observed[:, None] * (self.A @ mean + self.noise.mean_flat)
        cholesky_new = np.abs(self.to_observed[:, None]) * cholesky_new
        return IsotropicNormal(mean_new, cholesky_new, self.noise.tree_flatten)

    def merge(self, other: "IsotropicLatentCond", /) -> "IsotropicLatentCond":
        # self = cond1 (outer), other = cond2 (inner)
        T = self.to_latent * other.to_observed

        g = self.A @ (T[:, None] * other.A)
        xi = self.A @ (T[:, None] * other.noise.mean_flat) + self.noise.mean_flat

        R1 = (self.A @ (np.abs(T[:, None]) * other.noise.cholesky_flat)).T
        R2 = self.noise.cholesky_flat.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        noise = IsotropicNormal(xi, Xi.T, self.noise.tree_flatten)
        return IsotropicLatentCond(
            g, noise, to_latent=other.to_latent, to_observed=self.to_observed
        )

    def revert(self, rv, /, *, solve_triu):
        mean = self.to_latent[:, None] * rv.mean_flat
        cholesky = np.abs(self.to_latent[:, None]) * rv.cholesky_flat

        R_X_F = (self.A @ cholesky).T
        R_X = cholesky.T
        R_YX = self.noise.cholesky_flat.T
        tmp = cholesky_util.revert_conditional(
            R_X_F=R_X_F, R_X=R_X, R_YX=R_YX, solve_triu=solve_triu
        )
        r_obs, (r_cor, gain) = tmp

        mean_observed = self.A @ mean + self.noise.mean_flat
        mean_corrected = mean - gain @ mean_observed
        cholesky_corrected = r_cor.T
        corrected = IsotropicNormal(mean_corrected, cholesky_corrected, rv.tree_flatten)
        cond_new = IsotropicLatentCond(
            gain,
            corrected,
            to_latent=1 / self.to_observed,
            to_observed=1 / self.to_latent,
        )

        mean = self.to_observed[:, None] * mean_observed
        cholesky = np.abs(self.to_observed[:, None]) * r_obs.T
        observed = IsotropicNormal(mean, cholesky, self.noise.tree_flatten)
        return observed, cond_new

    def preconditioner_apply(self, /):
        A = self.to_observed[:, None] * self.A * self.to_latent[None, :]
        mean = self.to_observed[:, None] * self.noise.mean_flat
        cholesky = np.abs(self.to_observed[:, None]) * self.noise.cholesky_flat
        noise = IsotropicNormal(mean, cholesky, self.noise.tree_flatten)
        return IsotropicLatentCond.from_linop_and_noise(A, noise)


IsotropicLatentCond._register_as_pytree()


class IsotropicNormal(ssm_impl_api.AbstractTreeNormal[IsotropicTreeFlatten]):
    """Construct an isotropic normal distribution."""

    @classmethod
    def from_dirac(cls, mean, *, damp):
        utilities.verify_taylor_coefficient_pytree(mean)

        def is_leaf(s):
            return tree.tree_structure(s) == tree.tree_structure(mean[0])

        leaves, structure = tree.tree_flatten_depth_one(mean)
        leaves_ = [np.asarray(damp) for _ in leaves]
        std = tree.tree_unflatten(structure, leaves_)
        return IsotropicNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        utilities.verify_taylor_coefficient_pytree(mean)
        utilities.verify_taylor_coefficient_pytree(std)

        tree_flatten = IsotropicTreeFlatten.from_example(mean)
        loc_flat = tree_flatten.flatten_tree(mean)
        scale_flat = tree_flatten.flatten_tree_scalar(std)

        num_coeffs = len(mean)
        if scale_flat.shape != (num_coeffs,):
            msg = "'std' must have the same pytree structure as mean, "
            msg += "but each leaf must be a scalar instead of an array. "
            msg += f"Received: {std}"
            raise ValueError(msg)

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return cls(loc_flat, cholesky_flat, tree_flatten)

    @property
    def mean(self):
        return self._mean_batched()

    def _mean_batched(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(IsotropicNormal._mean_batched)(self)
        return self.tree_flatten.unflatten_array(self.mean_flat)

    @property
    def std(self):
        return self._std_batched()

    def _std_batched(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(IsotropicNormal._std_batched)(self)
        std_flat = func.vmap(linalg.vector_norm)(self.cholesky_flat)
        return self.tree_flatten.unflatten_array_scalar(std_flat)

    def residual_whitened_rms_tree(self, u):
        if self.cholesky_flat.size > 1:
            raise ValueError
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.residual_whitened_rms_flat(u_latent)

    def residual_whitened_rms_flat(self, u_latent, /):
        residual_white = (self.mean_flat - u_latent) / self.cholesky_flat
        residual_white_matrix = linalg.qr_r(residual_white.T)
        return np.reshape(
            np.abs(residual_white_matrix) / np.sqrt(self.mean_flat.size), ()
        )

    def rescale_cholesky(self, factor, /):
        cholesky = factor[..., None, None] * self.cholesky_flat
        return IsotropicNormal(self.mean_flat, cholesky, self.tree_flatten)

    def logpdf_tree(self, u, /):
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.logpdf_flat(u_latent)

    def logpdf_flat(self, u_latent, /):
        # Batch in the "mean" dimension and sum the results.
        in_axes = (IsotropicNormal(1, None, self.tree_flatten), 1)
        logpdf_vmap = func.vmap(IsotropicNormal.logpdf_scalar_flat, in_axes=in_axes)

        logpdfs = logpdf_vmap(self, u_latent)
        return np.sum(logpdfs)

    def logpdf_scalar_flat(self, u, /):
        cholesky = linalg.qr_r(self.cholesky_flat.T).T

        dx = u - self.mean_flat
        w = linalg.solve_triu(cholesky.T, dx, trans="T")

        maha_term = linalg.vector_dot(w, w)

        diagonal = linalg.diagonal_along_axis(cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * np.log(np.pi() * 2))

    def to_multivariate_normal(self):
        _n, d = self.mean_flat.shape
        eye_d = np.eye(d)

        cov = self.cholesky_flat @ self.cholesky_flat.T

        cov = np.kron(eye_d, cov)
        mean = self.mean_flat.reshape((-1,), order="F")
        return (mean, cov)

    def sample_tree(self, key):
        sample_latent = self.sample_flat(key)
        return self.tree_flatten.unflatten_array(sample_latent)

    def sample_flat(self, key):
        n, _n = self.cholesky_flat.shape
        base = random.normal(key, shape=(n,))
        return self.mean_flat + (self.cholesky_flat @ base)[:, None]

    def identity_conditional(self) -> IsotropicLatentCond:
        num, d = self.mean_flat.shape
        m0 = np.zeros((num, d))
        c0 = np.zeros((num, num))
        noise = IsotropicNormal(m0, c0, self.tree_flatten)
        matrix = np.eye(num)
        return IsotropicLatentCond.from_linop_and_noise(matrix, noise)

    def prototype_output_scale_calibrated(self):
        return np.ones(())

    def to_derivative(self, i, std):
        ndim, _d = self.mean_flat.shape
        m = np.zeros((ndim,))
        linop = func.jacfwd(lambda s: np.asarray([s[i]]))(m)

        u_like = tree.tree_map(np.zeros_like, self.mean[0])
        noise = IsotropicNormal.from_mean_and_std([u_like], [std])
        return IsotropicLatentCond.from_linop_and_noise(linop, noise)

    @staticmethod
    def register_pytree_node() -> None:
        def flatten(normal):
            children = normal.mean_flat, normal.cholesky_flat
            aux = (normal.tree_flatten,)
            return children, aux

        def unflatten(aux, children):
            (tree_flatten,) = aux
            mean, cholesky = children
            return IsotropicNormal(mean, cholesky, tree_flatten)

        tree.register_pytree_node(IsotropicNormal, flatten, unflatten)


IsotropicNormal.register_pytree_node()


class IsotropicOdeTs0(ssm_impl_api.AbstractOde):
    """Isotropic ODE linearization via TS0 (zeroth-degree Taylor series: evaluate at the prior mean, no Jacobian)."""

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv, state: None, *, damp: float, t):
        del state
        Ms = rv.mean
        jet_coords = Ms[: self.ode.num_tcoeffs_in_args]
        fx_tree = self.ode.vector_field(jet_coords=jet_coords, t=t)
        fx = tree.tree_map(lambda s: -s, fx_tree)

        bias = IsotropicNormal.from_dirac([fx], damp=damp)

        linop = func.jacrev(lambda s: s[[self.ode.num_tcoeffs_in_args], ...])(
            rv.mean_flat[..., 0]
        )
        cond = IsotropicLatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class IsotropicResidual(ssm_impl_api.AbstractResidual):
    """Isotropic residual linearization via TS1."""

    def init_linearization(self):
        return self.residual.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t):

        def residual_flat(s_stack):
            s_tree = rv.tree_flatten.unflatten_array(s_stack)
            s_tree = s_tree[: self.residual.num_tcoeffs_in_args]
            fs = self.residual.residual_function(jet_coords=s_tree, t=t)
            return tree.ravel_pytree(fs)[0][None, :]

        _n, d = rv.mean_flat.shape

        fx, J_trace, state = self.residual.jacobian.calculate_trace_along_d(
            residual_flat, rv.mean_flat, state
        )
        J_trace /= d

        linop = J_trace
        fx = fx - linop @ rv.mean_flat

        m0_tree = rv.mean[:1]
        rv0 = IsotropicNormal.from_dirac(m0_tree, damp=0.0)
        fx = rv0.tree_flatten.unflatten_array(fx)

        noise = IsotropicNormal.from_dirac(fx, damp=damp)
        cond = IsotropicLatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class IsotropicWienerIntegrated(ssm_impl_api.AbstractPrior):
    def __init__(self, init, output_scale, *, A, q_sqrtm, q0, tree_flatten, precon_fun):
        super().__init__(init, output_scale)
        self.A = A
        self.q_sqrtm = q_sqrtm
        self.q0 = q0
        self.tree_flatten = tree_flatten
        self.precon_fun = precon_fun

    def transition(self, dt, output_scale: Array = 1.0):
        output_scale = np.asarray(output_scale)
        if output_scale.shape != ():
            msg = "The base-scale has the wrong shape."
            msg += f" Expected: {()}."
            msg += f" Received: {output_scale.shape}."
            raise ValueError(msg)

        scale = np.sqrt(np.abs(dt)) * self.output_scale * output_scale
        noise = IsotropicNormal(self.q0, scale * self.q_sqrtm, self.tree_flatten)
        p, p_inv = self.precon_fun(dt)
        return IsotropicLatentCond(self.A, noise, to_latent=p_inv, to_observed=p)

    @staticmethod
    def register_pytree():
        def flatten(iwp):
            children = (iwp.init, iwp.output_scale, iwp.A, iwp.q_sqrtm, iwp.q0)
            aux = (iwp.tree_flatten, iwp.precon_fun)
            return children, aux

        def unflatten(aux, children):
            tf, precon_fun = aux
            init, output_scale, A, q_sqrtm, q0 = children
            return IsotropicWienerIntegrated(
                init,
                output_scale,
                A=A,
                q_sqrtm=q_sqrtm,
                q0=q0,
                tree_flatten=tf,
                precon_fun=precon_fun,
            )

        tree.register_pytree_node(IsotropicWienerIntegrated, flatten, unflatten)


IsotropicWienerIntegrated.register_pytree()


class state_space_model_isotropic(ssm_impl_api.StateSpaceModel):
    """Isotropic (scalar-variance) state-space model implementation."""

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
        init = IsotropicNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        if output_scale is None:
            output_scale = np.ones(())
        else:
            if tree.tree_structure(output_scale) != tree.tree_structure(1.0):
                msg = "The 'base_scale' argument has an unexpected PyTree structure."
                msg += f" Expected: {tree.tree_structure(1.0)}."
                msg += f" Received: {tree.tree_structure(output_scale)}."
                raise TypeError(msg)

            output_scale = np.asarray(output_scale)
            if output_scale.shape != ():
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {()}."
                msg += f" Received: {output_scale.shape}."
                raise ValueError(msg)

        num_derivatives = len(tcoeffs_mean) - 1
        (d,) = tree.ravel_pytree(tcoeffs_mean[0])[0].shape
        A, q_sqrtm = utilities.system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1, d))
        tf = IsotropicTreeFlatten.from_example(tcoeffs_mean)
        precon_fun = utilities.preconditioner_taylor(num_derivatives)
        return IsotropicWienerIntegrated(
            init,
            output_scale,
            A=A,
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
        msg = "Isotropic exponential priors have not been implemented (yet.)."
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
        msg = "Isotropic exponential priors have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def _tcoeffs_standard_deviation(self, tcoeffs_mean, /, *, is_exact, inexact_eps):

        leaves, structure = tree.tree_flatten_depth_one(tcoeffs_mean)
        std_template = tree.tree_unflatten(structure, [np.zeros(()) for _ in leaves])

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: np.shape(a) == np.shape(b), A, B)

        # Construct the initial std.
        # If is_exact is a boolean, copy the pytree structure from the mean
        # Otherwise, set the initial std element-wise.
        if isinstance(is_exact, bool):
            if is_exact:
                tcoeffs_std = tree.tree_map(np.zeros_like, std_template)
            else:

                def eps_like(s):
                    return inexact_eps * np.ones_like(s)

                tcoeffs_std = tree.tree_map(eps_like, std_template)
        else:
            if not tree.tree_all(shape_equal(is_exact, std_template)):
                msg = "Input 'is_exact' has the wrong PyTree structure."
                msg += f" Expected: {tree.tree_map(np.shape, std_template)}."
                msg += f" Received: {tree.tree_map(np.shape, is_exact)}."
                raise ValueError(msg)

            def std_init(s: Array) -> Array:
                if s.dtype != np.dtype(bool):
                    msg = "Boolean entries expected in `is_exact`."
                    msg += f" Received: dtype={np.dtype(s)}"
                    raise TypeError(msg)
                return np.where(s, 0.0, inexact_eps)

            tcoeffs_std = tree.tree_map(std_init, is_exact)
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

    def constraint_ode_ts0(self, ode: problems.JetOde, /) -> IsotropicOdeTs0:
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)
        return IsotropicOdeTs0(ode=ode)

    def constraint_residual(
        self,
        residual: problems.JetResidual,
        *,
        taylor_point: taylor_points.TaylorPoint | None = None,
    ) -> IsotropicResidual:
        if not isinstance(residual, problems.JetResidual):
            raise TypeError(residual)
        if taylor_point is not None:
            raise NotImplementedError
        return IsotropicResidual(residual)
