from probdiffeq._ssm_util import ssm_api, utilities
from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Callable, Literal, Sequence, TypeVar
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

__all__ = [
    "IsotropicConditional",
    "IsotropicLinearizationFactory",
    "IsotropicNormal",
    "IsotropicOdeTs0",
    "IsotropicOdeTs1",
    "IsotropicPriorFactory",
    "IsotropicTreeFlatten",
]


@structs.dataclass
class IsotropicTreeFlatten(ssm_api.AbstractTreeFlatten):
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


class IsotropicNormal(ssm_api.AbstractTreeNormal[IsotropicTreeFlatten]):
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
        # TODO: use QR decomp instead of einsum to avoid sqrts
        diag = np.einsum("ij,ji->i", self.cholesky_flat, self.cholesky_flat)
        std_flat = np.sqrt(diag)
        return self.tree_flatten.unflatten_array_scalar(std_flat)

    def residual_white_rms_tree(self, u):
        if self.cholesky_flat.size > 1:
            raise ValueError
        u_latent = self.tree_flatten.flatten_tree(u)
        return self.residual_white_rms_flat(u_latent)

    def residual_white_rms_flat(self, u_latent, /):
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


class IsotropicPriorFactory(ssm_api.AbstractPriorFactory):
    """Implementation of isotropic prior constructors."""

    def identity(self, template) -> ssm_api.LatentCond:
        num, d = template.mean_flat.shape
        m0 = np.zeros((num, d))
        c0 = np.zeros((num, num))
        noise = IsotropicNormal(m0, c0, template.tree_flatten)
        matrix = np.eye(num)
        return ssm_api.LatentCond.from_linop_and_noise(matrix, noise)

    def wiener_integrated(
        self,
        tcoeffs_mean: C,
        /,
        *,
        is_exact: C | bool,
        inexact_eps: float,
        diffuse_derivatives: int,
        diffuse_eps: float,
        base_scale: Array | None,
    ):
        tcoeffs_std = self._tcoeffs_standard_deviation(
            tcoeffs_mean, is_exact=is_exact, inexact_eps=inexact_eps
        )
        return self.wiener_integrated_diffuse(
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            base_scale=base_scale,
        )

    def wiener_integrated_diffuse(
        self,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int,
        diffuse_eps: float,
        base_scale: Array | None,
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

        if base_scale is None:
            base_scale = np.ones(())
        else:
            base_scale = np.asarray(base_scale)
            if base_scale.shape != ():
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {()}."
                msg += f" Received: {base_scale.shape}."
                raise ValueError(msg)

        num_derivatives = len(tcoeffs_mean) - 1
        (d,) = tree.ravel_pytree(tcoeffs_mean[0])[0].shape
        A, q_sqrtm = utilities.system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1, d))
        precon_fun = utilities.preconditioner_taylor(num_derivatives=num_derivatives)

        def discretise(dt, output_scale: Array = 1.0):
            output_scale = np.asarray(output_scale)
            if output_scale.shape != ():
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {()}."
                msg += f" Received: {output_scale.shape}."
                raise ValueError(msg)

            scale = base_scale * output_scale

            p, p_inv = precon_fun(dt)
            tree_flatten = IsotropicTreeFlatten.from_example(tcoeffs_mean)
            noise = IsotropicNormal(q0, scale * q_sqrtm, tree_flatten)

            return ssm_api.LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        return init, discretise

    def exponential(
        self,
        tcoeffs_mean: C,
        /,
        *,
        vf_linear,
        is_exact: C | bool,
        inexact_eps: float,
        diffuse_derivatives: int,
        diffuse_eps: float,
        base_scale: Array | None,
    ):
        del tcoeffs_mean
        del vf_linear
        del is_exact
        del inexact_eps
        del diffuse_derivatives
        del diffuse_eps
        del base_scale
        msg = "Isotropic exponential priors have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def exponential_diffuse(
        self,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        vf_linear,
        diffuse_derivatives: int,
        diffuse_eps: float,
        base_scale: Array | None,
    ):
        del tcoeffs_mean
        del tcoeffs_std
        del vf_linear
        del diffuse_derivatives
        del diffuse_eps
        del base_scale
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

    def to_derivative(self, i, std, template):

        ndim, _d = template.mean_flat.shape
        m = np.zeros((ndim,))
        linop = func.jacfwd(lambda s: np.asarray([s[i]]))(m)

        u_like = tree.tree_map(np.zeros_like, template.mean[0])

        # Wrap u_like and std into a list because the random variable
        # expects TaylorCoefficients.
        noise = IsotropicNormal.from_mean_and_std([u_like], [std])
        return ssm_api.LatentCond.from_linop_and_noise(linop, noise)

    def prototype_output_scale_calibrated(self, template):
        del template
        return np.ones(())


class IsotropicOdeTs0(ssm_api.AbstractOde):
    """Construct an isotropic implementation of ODE-TS0 linearization."""

    def __init__(self, vf, *, ode_order) -> None:
        super().__init__(vf, ode_order=ode_order)

    @property
    def root_order(self):
        return self.ode_order + 1

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv, state: None, *, damp: float, t):
        del state
        fun = func.partial(self.vector_field, t=t)
        Ms = rv.mean
        fx_tree = fun(*(Ms[: self.ode_order]))
        fx = tree.tree_map(lambda s: -s, fx_tree)

        bias = IsotropicNormal.from_dirac([fx], damp=damp)

        linop = func.jacrev(lambda s: s[[self.ode_order], ...])(rv.mean_flat[..., 0])
        cond = ssm_api.LatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class IsotropicOdeTs1(ssm_api.AbstractOde):
    """Construct an isotropic implementation of ODE-TS1 linearization."""

    def __init__(self, vf, *, ode_order: int, jacobian: Any) -> None:
        if ode_order > 1:
            msg = "This linearization is not compatible with high-order ODEs as of yet."
            raise ValueError(msg)
        super().__init__(vf, ode_order=1)

        self.jacobian = jacobian

    def init_linearization(self):
        return self.jacobian.init_jacobian_handler()

    def linearize(self, rv, state, *, damp: float, t):
        fun = func.partial(self.vector_field, t=t)
        # Evaluate the linearisation
        m0 = rv.mean_flat[0]

        m0_tree = rv.mean[0]
        rv0 = IsotropicNormal.from_dirac([m0_tree], damp=0.0)

        def vf_ravel(s):
            s_tree = rv0.tree_flatten.unflatten_array(s[None])
            fs = fun(*s_tree)
            return tree.ravel_pytree(fs)[0]

        # Estimate the trace using Hutchinson's estimator
        fx, J_trace, state = self.jacobian.calculate_trace(vf_ravel, m0, state)

        # Best Jacobian approximation: mean of diagonal = trace / len(diagonal)
        J_trace /= len(fx)

        n, _d = rv.mean_flat.shape
        eye = np.eye(n)
        E0, E1 = eye[np.asarray([0])], eye[np.asarray([1])]
        linop = E1 - J_trace * E0
        fx = rv.mean_flat[1, ...] - fx
        fx = fx - linop @ rv.mean_flat
        fx = rv0.tree_flatten.unflatten_array(fx)

        # Turn fx and J_trace into an observation model
        noise = IsotropicNormal.from_dirac(fx, damp=damp)
        cond = ssm_api.LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class IsotropicLinearizationFactory(ssm_api.AbstractLinearizationFactory):
    """Construct an isotropic linearization-factory."""

    def root(
        self,
        root,
        *,
        jacobian,
        root_order: int | Literal["max"],
        nlstsq: Callable | None,
    ):
        raise NotImplementedError

    def ode_taylor_1st(self, vf, *, ode_order, jacobian):
        return IsotropicOdeTs1(vf, jacobian=jacobian, ode_order=ode_order)

    def ode_taylor_0th(self, vf, *, ode_order):
        return IsotropicOdeTs0(vf, ode_order=ode_order)


class IsotropicConditional(ssm_api.AbstractConditional):
    """Construct an isotropic implementation of manipulating conditionals."""

    def apply_flat(self, x, cond, /):
        x = cond.to_latent[:, None] * x
        mean_new = cond.to_observed[:, None] * cond.A @ x + cond.noise.mean_flat
        cholesky_new = cond.to_observed[:, None] * cond.noise.cholesky_flat
        return IsotropicNormal(mean_new, cholesky_new, cond.noise.tree_flatten)

    def marginalise(self, rv, cond, /):
        mean = cond.to_latent[:, None] * rv.mean_flat
        cholesky = cond.to_latent[:, None] * rv.cholesky_flat
        R_stack = ((cond.A @ cholesky).T, cond.noise.cholesky_flat.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = cond.to_observed[:, None] * (cond.A @ mean + cond.noise.mean_flat)
        cholesky_new = cond.to_observed[:, None] * cholesky_new
        return IsotropicNormal(mean_new, cholesky_new, cond.noise.tree_flatten)

    def merge(self, cond1, cond2, /):
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        g = cond1.A @ (T[:, None] * cond2.A)

        # Combined mean
        xi = cond1.A @ (T[:, None] * cond2.noise.mean_flat) + cond1.noise.mean_flat

        # Cholesky factor of combined covariance
        R1 = (cond1.A @ (T[:, None] * cond2.noise.cholesky_flat)).T
        R2 = cond1.noise.cholesky_flat.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        # Gather and return
        noise = IsotropicNormal(xi, Xi.T, cond1.noise.tree_flatten)
        return ssm_api.LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv, cond, /, *, solve_triu):
        # Pull RV into the latent space
        mean = cond.to_latent[:, None] * rv.mean_flat
        cholesky = cond.to_latent[:, None] * rv.cholesky_flat

        # QR-decomposition
        R_X_F, R_X, R_YX = (cond.A @ cholesky).T, cholesky.T, cond.noise.cholesky_flat.T
        tmp = cholesky_util.revert_conditional(
            R_X_F=R_X_F, R_X=R_X, R_YX=R_YX, solve_triu=solve_triu
        )
        r_obs, (r_cor, gain) = tmp

        # Push correction into observed space
        mean_observed = cond.A @ mean + cond.noise.mean_flat
        mean_corrected = mean - gain @ mean_observed
        cholesky_corrected = r_cor.T
        corrected = IsotropicNormal(mean_corrected, cholesky_corrected, rv.tree_flatten)
        cond_new = ssm_api.LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather the observed variable
        mean = cond.to_observed[:, None] * mean_observed
        cholesky = cond.to_observed[:, None] * r_obs.T
        observed = IsotropicNormal(mean, cholesky, cond.noise.tree_flatten)
        return observed, cond_new

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[:, None] * cond.A * cond.to_latent[None, :]
        mean = cond.to_observed[:, None] * cond.noise.mean_flat
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky_flat
        noise = IsotropicNormal(mean, cholesky, cond.noise.tree_flatten)
        return ssm_api.LatentCond.from_linop_and_noise(A, noise)


IsotropicNormal.register_pytree_node()
