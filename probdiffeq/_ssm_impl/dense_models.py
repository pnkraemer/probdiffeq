from probdiffeq._ssm_util import ssm_api, utilities
from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Callable, Literal, Sequence, TypeVar
from probdiffeq.util import cholesky_util, gram_util

__all__ = [
    "DenseConditional",
    "DenseLinearizationFactory",
    "DenseNormal",
    "DenseOdeTs0",
    "DenseOdeTs1",
    "DensePriorFactory",
    "DenseRoot",
    "DenseTreeFlatten",
]

C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""


class DensePriorFactory(ssm_api.AbstractPriorFactory):
    """Implementation of dense prior constructors."""

    def identity(self, template) -> ssm_api.LatentCond:
        (n,) = template.mean_flat.shape
        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        noise = DenseNormal(m, C, template.tree_flatten)
        return ssm_api.LatentCond.from_linop_and_noise(A, noise)

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
        init = DenseNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        # Process the base-scale
        single_flat, single_unravel = tree.ravel_pytree(tcoeffs_mean[0])
        Lambda = self._process_base_scale(base_scale, single_flat, single_unravel)

        # Construct the transitions
        num_derivatives = len(tcoeffs_mean) - 1
        a, q_sqrtm = utilities.system_matrices_1d_iwp(num_derivatives)
        (d,) = single_flat.shape
        eye_d = np.eye(d)
        A = np.kron(a, eye_d)
        Q = np.kron(q_sqrtm, Lambda)

        q0 = np.zeros(((num_derivatives + 1) * d,))
        precon_fun = utilities.preconditioner_taylor(num_derivatives=num_derivatives)

        def discretise(dt, output_scale=1.0):
            output_scale = self._process_calibrated_scale(output_scale)

            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            tree_flatten = DenseTreeFlatten.from_example(tcoeffs_mean)
            noise = DenseNormal(q0, output_scale * Q, tree_flatten)
            return ssm_api.LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        # Return the initial variable and the discretisation
        return init, discretise

    def exponential(
        self,
        tcoeffs_mean: C,
        /,
        *,
        vf_linear: Array,
        is_exact: C | bool,
        inexact_eps: float,
        diffuse_derivatives: int,
        diffuse_eps: float,
        base_scale: Array | None,
    ):
        tcoeffs_std = self._tcoeffs_standard_deviation(
            tcoeffs_mean, is_exact=is_exact, inexact_eps=inexact_eps
        )
        return self.exponential_diffuse(
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            base_scale=base_scale,
            vf_linear=vf_linear,
        )

    def exponential_diffuse(
        self,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        vf_linear: Array,
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
        init = DenseNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        # Process the base-scale
        single_flat, single_unravel = tree.ravel_pytree(tcoeffs_mean[0])
        Lambda = self._process_base_scale(base_scale, single_flat, single_unravel)

        # Turn the linear vector field into the bottom block of the IOUP
        # by building the matrix-version of the vector field's Jacobian.

        # First, set up a template variable

        leaves_flat, unflatten = tree.ravel_pytree(tcoeffs_mean)

        def vf_flat(tcoeffs_flat):
            tcoeffs_tree = unflatten(tcoeffs_flat)
            fx = vf_linear(*tcoeffs_tree)
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

        precon_fun = utilities.preconditioner_taylor(num_derivatives=num_derivatives)

        # TODO: find a good default. Always using high orders seems wasteful.
        pade_legendre = (
            gram_util.pade_and_legendre_9()
            if B.dtype == "float64"
            else gram_util.pade_and_legendre_5()
        )

        # Pascal matrices are upper triangular so we use a dedicated solver
        exp_gram = gram_util.exp_gram_cholesky(
            pade_legendre=pade_legendre, solve=linalg.solve_lu
        )
        q0 = np.zeros(leaves_flat.shape)

        # TODO: why is there a default argument for output_scale?
        def discretise(dt, output_scale=1.0):
            output_scale = self._process_calibrated_scale(output_scale)

            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            # Precondition. (I've not seen a big impact, but we do it anyway)
            A_p = dt * p_inv[:, None] * A * p[None, :]
            B_p = np.sqrt(dt) * p_inv[:, None] * B

            eA, L = exp_gram(A_p, B_p)
            tree_flatten = DenseTreeFlatten.from_example(tcoeffs_mean)
            noise = DenseNormal(q0, output_scale * L, tree_flatten)
            return ssm_api.LatentCond(eA, noise, to_latent=p_inv, to_observed=p)

        return init, discretise

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
        base_scale = np.asarray(base_scale)
        if base_scale.shape != base_scale_expected.shape:
            msg = "The base-scale has the wrong shape."
            msg += f" Expected: {base_scale_expected.shape}."
            msg += f" Received: {base_scale.shape}."
            raise ValueError(msg)

        # Flatten the scale into something compatible with the flattened SSM
        base_scale, _ = tree.ravel_pytree(base_scale)
        return linalg.diagonal_matrix(base_scale)

    def _process_calibrated_scale(self, output_scale):
        output_scale = np.asarray(output_scale)
        if output_scale.shape != ():
            msg = "The output-scale has the wrong shape."
            msg += f" Expected: {()}."
            msg += f" Received: {output_scale.shape}."
            raise ValueError(msg)
        return output_scale

    def to_derivative(self, i, std, template):
        all_flat, all_unravel = tree.ravel_pytree(template.mean)

        def select(a):
            return tree.ravel_pytree(all_unravel(a)[i])[0]

        x = np.zeros(all_flat.shape)
        linop = func.jacfwd(select)(x)

        data_like = all_unravel(x)[0]
        noise = DenseNormal.from_mean_and_std([data_like], [std])
        return ssm_api.LatentCond.from_linop_and_noise(linop, noise)

    def prototype_output_scale_calibrated(self, template):
        del template
        return np.ones(())


@structs.dataclass
class DenseTreeFlatten(ssm_api.AbstractTreeFlatten):
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


class DenseNormal(ssm_api.AbstractTreeNormal[DenseTreeFlatten]):
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

    def residual_white_rms_tree(self, u):
        u, _ = tree.ravel_pytree(u)
        return self.residual_white_rms_flat(u)

    def residual_white_rms_flat(self, u):
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


class DenseLinearizationFactory(ssm_api.AbstractLinearizationFactory):
    """Construct a dense linearization factory."""

    def root(self, root, *, jacobian, root_order: int | Literal["max"], nlstsq: bool):
        return DenseRoot(root, jacobian=jacobian, root_order=root_order, nlstsq=nlstsq)

    def ode_taylor_0th(self, vf, *, ode_order):
        return DenseOdeTs0(vf, ode_order=ode_order)

    def ode_taylor_1st(self, vf, *, ode_order, jacobian):
        if ode_order > 1:
            raise ValueError

        return DenseOdeTs1(vf, ode_order=ode_order, jacobian=jacobian)


class DenseConditional(ssm_api.AbstractConditional):
    """Construct a dense implementation of manipulating conditionals."""

    def apply_flat(self, x, cond, /):
        x = cond.to_latent * x
        mean = cond.to_observed * (cond.A @ x + cond.noise.mean_flat)
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky_flat
        return DenseNormal(mean, cholesky, cond.noise.tree_flatten)

    def marginalise(self, rv, cond, /):
        mean = cond.to_latent * rv.mean_flat
        cholesky = cond.to_latent[:, None] * rv.cholesky_flat

        R_stack = ((cond.A @ cholesky).T, cond.noise.cholesky_flat.T)
        cholesky_new = cholesky_util.sum_of_sqrtm_factors(R_stack=R_stack).T

        mean_new = cond.to_observed * (cond.A @ mean + cond.noise.mean_flat)
        cholesky_new = cond.to_observed[:, None] * cholesky_new
        return DenseNormal(mean_new, cholesky_new, cond.noise.tree_flatten)

    def merge(
        self, cond1: ssm_api.LatentCond, cond2: ssm_api.LatentCond, /
    ) -> ssm_api.LatentCond:
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        g = cond1.A @ (T[:, None] * cond2.A)

        # Combined mean
        xi = cond1.A @ (T * cond2.noise.mean_flat) + cond1.noise.mean_flat

        # Cholesky factor of combined covariance
        R1 = (cond1.A @ (T[:, None] * cond2.noise.cholesky_flat)).T
        R2 = cond1.noise.cholesky_flat.T
        Xi = cholesky_util.sum_of_sqrtm_factors(R_stack=(R1, R2))

        # Gather and return
        noise = DenseNormal(xi, Xi.T, cond1.noise.tree_flatten)
        return ssm_api.LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(
        self, rv: DenseNormal, cond: ssm_api.LatentCond, /, *, solve_triu: Callable
    ):
        # Pull RV into the latent space
        mean = cond.to_latent * rv.mean_flat
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
        corrected = DenseNormal(mean_corrected, cholesky_corrected, rv.tree_flatten)
        cond_new = ssm_api.LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather the observed variable
        mean = cond.to_observed * mean_observed
        cholesky = cond.to_observed[:, None] * r_obs.T
        observed = DenseNormal(mean, cholesky, cond.noise.tree_flatten)
        return observed, cond_new

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[:, None] * cond.A * cond.to_latent[None, :]
        mean = cond.to_observed * cond.noise.mean_flat
        cholesky = cond.to_observed[:, None] * cond.noise.cholesky_flat
        noise = DenseNormal(mean, cholesky, cond.noise.tree_flatten)
        return ssm_api.LatentCond.from_linop_and_noise(A, noise)


class DenseOdeTs0(ssm_api.AbstractOde):
    """Construct a dense implementation of ODE-TS0 linearization."""

    def __init__(self, vf, *, ode_order: int) -> None:
        super().__init__(vf, ode_order=ode_order)

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv: DenseNormal, state: None, *, damp: float, t):
        fun = func.partial(self.vector_field, t=t)
        del state

        def a1(m: Array) -> Array:
            """Select the 'n'-th derivative."""
            m0 = rv.tree_flatten.unflatten_array(m)[self.ode_order]
            return tree.ravel_pytree(m0)[0]

        Ms = rv.mean

        fm = fun(*Ms[: self.ode_order])
        fx = tree.tree_map(lambda s: -s, [fm])
        linop = func.jacrev(a1)(rv.mean_flat)
        noise = DenseNormal.from_dirac(fx, damp=damp)
        cond = ssm_api.LatentCond.from_linop_and_noise(linop, noise)
        return cond, None


class DenseOdeTs1(ssm_api.AbstractOde):
    """Construct a dense implementation of ODE-TS1 linearization."""

    def __init__(self, vf: Callable, ode_order: int, jacobian: Any) -> None:
        if ode_order > 1:
            msg = "Not implemented. Try the a root-based TS1 constraint instead."
            raise ValueError(msg)
        super().__init__(vf, ode_order=ode_order)
        self.jacobian = jacobian

    @property
    def root_order(self):
        return self.ode_order + 1

    def init_linearization(self):
        return self.jacobian.init_jacobian_handler()

    def linearize(self, rv, state: None, *, damp: float, t):
        fun = func.partial(self.vector_field, t=t)
        m_tree = rv.mean

        rv0 = DenseNormal.from_dirac([m_tree[0]], damp=0.0)

        def vf_flat(s: Array) -> Array:
            s0 = rv0.tree_flatten.unflatten_array(s)
            fs0 = fun(*s0)
            return rv0.tree_flatten.flatten_tree([fs0])

        def select_i(i) -> Callable[[Array], Array]:
            def select(s: Array) -> Array:
                s_tree = rv.tree_flatten.unflatten_array(s)
                return rv0.tree_flatten.flatten_tree(s_tree[i])

            return select

        E0 = func.jacfwd(select_i(i=0))(rv.mean_flat)
        E1 = func.jacfwd(select_i(i=1))(rv.mean_flat)

        m0 = rv0.mean_flat
        fx, J, state = self.jacobian.materialize_dense(vf_flat, m0, state)
        linop = E1 - J @ E0
        fx = -(fx - J @ m0)
        fx = rv0.tree_flatten.unflatten_array(fx)
        noise = DenseNormal.from_dirac(fx, damp=damp)
        cond = ssm_api.LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class DenseRoot(ssm_api.AbstractRoot):
    """Construct a dense implementation of root-TS1 linearization."""

    def __init__(
        self, root, *, root_order: int | Literal["max"], jacobian, nlstsq
    ) -> None:
        super().__init__(root, root_order=root_order)
        self.jacobian = jacobian
        self.nlstsq = nlstsq

    def init_linearization(self):
        return self.jacobian.init_jacobian_handler()

    def constraint_flat(self, m: Array, *, t, tree_flatten) -> Array:
        """Evaluate a flattened version of the root constraint."""
        # Unravel the location and extract derivatives
        m_tree = tree_flatten.unflatten_array(m)
        relevant_tcoeffs = (
            m_tree if self.root_order == "max" else m_tree[: self.root_order]
        )

        # Evaluate the root
        root_eval = self.root(*relevant_tcoeffs, t=t)

        # Flatten the output so that the Jacobians are matrices, not Pytrees.
        return tree.ravel_pytree(root_eval)[0]

    def linearize(self, rv, state, *, damp: float, t):

        # Fix all arguments except the Array ones
        constraint_flat = func.partial(
            self.constraint_flat, t=t, tree_flatten=rv.tree_flatten
        )

        # Initial guess for the linearization point

        mean = rv.mean_flat
        if self.nlstsq is not None:  # posterior linearization
            mean, _info = self.nlstsq(
                constraint_flat, mean, rv.mean_flat, rv.cholesky_flat
            )

        fx, linop, state = self.jacobian.materialize_dense(constraint_flat, mean, state)
        fx = fx - linop @ mean

        # Find the tree structure of the output constraint
        # (So that we can unravel the bias term and always work in the correct
        # pytree structure.)
        m_tree = rv.mean
        relevant_tcoeffs = (
            m_tree if self.root_order == "max" else m_tree[: self.root_order]
        )
        root_eval = func.eval_shape(lambda s: [self.root(*s, t=t)], relevant_tcoeffs)

        # Ensure that unravelling does not yield a ShapeDtypeStruct
        root_eval = tree.tree_map(np.zeros_like, root_eval)
        _, unravel = tree.ravel_pytree(root_eval)

        # Turn the linearization into a conditional
        noise = DenseNormal.from_dirac(unravel(fx), damp=damp)

        cond = ssm_api.LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


DenseNormal.register_pytree_node()
