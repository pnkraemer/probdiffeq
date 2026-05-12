"""Implementations of (factorized) state-space models."""

from probdiffeq.backend import abc, func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Array, Callable, Generic, Sequence, TypeVar
from probdiffeq.util import cholesky_util, gram_util

T = TypeVar("T", bound=Array)
"""A type-variable for Array types.

For example, this variable is used for means and Cholesky factors
in normal distributions.
"""

C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""


class LatentCond:
    """Conditional distributions in latent space."""

    def __init__(self, A, noise, to_latent, to_observed) -> None:
        self.A = A
        self.noise = noise
        self.to_latent = to_latent
        self.to_observed = to_observed

    def __repr__(self) -> str:
        msg = f"LatentCond(A={self.A}, noise={self.noise}"
        msg += f", to_latent={self.to_latent}, to_observed={self.to_observed})"
        return msg

    @staticmethod
    def register_pytree_node() -> None:
        """Register the conditional as a pytree."""

        def flatten(normal):
            children = normal.A, normal.noise, normal.to_latent, normal.to_observed
            return children, ()

        def unflatten(_aux, children):
            A, noise, to_latent, to_observed = children
            return LatentCond(A, noise, to_latent, to_observed)

        tree.register_pytree_node(LatentCond, flatten, unflatten)

    @classmethod
    def from_linop_and_noise(cls, A, noise):
        """Construct a latent conditional with unit en- and decoders."""
        # Hack for blockdiagonal models (and possibly dense evaluations)
        if A.ndim > 2:
            return func.vmap(cls.from_linop_and_noise)(A, noise)

        d_out, d_in = A.shape
        to_latent, to_observed = np.ones((d_in,)), np.ones((d_out,))

        return cls(A, noise=noise, to_latent=to_latent, to_observed=to_observed)

    def rescale_noise(self, factor, /):
        """Rescale the noise in a conditional."""
        noise = self.noise.rescale_cholesky(factor)
        return LatentCond(
            A=self.A,
            noise=noise,
            to_latent=self.to_latent,
            to_observed=self.to_observed,
        )


LatentCond.register_pytree_node()


class AbstractLinearization(abc.ABC):
    """Interface for linearizations."""

    @abc.abstractmethod
    def init_linearization(self):
        """Initialize a linearization."""
        raise NotImplementedError

    @abc.abstractmethod
    def linearize(self, rv, state: None, *, damp: float, t):
        """Evaluate a linearization."""
        raise NotImplementedError


class AbstractRoot(AbstractLinearization):
    """Interface for linearizations of general roots."""

    def __init__(self, root, /, *, root_order) -> None:
        self.root = root
        self.root_order = root_order


class AbstractOde(AbstractLinearization):
    """Interface for linearizations of ODEs."""

    def __init__(self, vf, /, *, ode_order) -> None:
        self.vector_field = vf
        self.ode_order = ode_order

    @property
    def root_order(self):
        """The order of the root constraint."""
        return self.ode_order + 1


class AbstractLinearizationFactory(abc.ABC):
    """Interface for linearization factories."""

    @abc.abstractmethod
    def root(
        self, root, *, jacobian, root_order: int, nlstsq: Callable | None
    ) -> AbstractRoot:
        """Construct an implementation of 1st-order Taylor-linearization for roots."""
        raise NotImplementedError

    @abc.abstractmethod
    def ode_taylor_0th(self, ode_order: int) -> AbstractOde:
        """Construct an implementation of 0th-order Taylor-linearization for ODEs."""
        raise NotImplementedError

    @abc.abstractmethod
    def ode_taylor_1st(self, vf, *, ode_order: int) -> AbstractOde:
        """Construct an implementation of 1st-order Taylor-linearization for ODEs."""
        raise NotImplementedError


class AbstractTreeFlatten(abc.ABC):
    """Abstract base class for flattening information of tree-structured random variables."""

    @abc.abstractmethod
    def flatten_tree(self, x):
        """Flatten a pytree into an array."""
        pass

    @abc.abstractmethod
    def unflatten_array(self, x):
        """Unflatten an array into a pytree."""
        pass


S = TypeVar("S", bound=AbstractTreeFlatten)
"""A type-variable for tree-flattening types.

Used to type normal distributions.
"""


class AbstractTreeNormal(abc.ABC, Generic[S]):
    """Interface for pytree-valued normal distributions."""

    def __init__(self, mean_flat: Array, cholesky_flat: Array, tree_flatten: S, /):
        self.mean_flat = mean_flat
        self.cholesky_flat = cholesky_flat
        self.tree_flatten = tree_flatten

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}({self.mean_flat}, {self.cholesky_flat}, {self.tree_flatten})"

    @abc.abstractmethod
    def mean_tree(self):
        """Evaluate the mean."""
        raise NotImplementedError

    @abc.abstractmethod
    def std_tree(self):
        """Evaluate the standard deviation."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_tree(self, key):
        """Sample from a normal distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_flat(self, key):
        """Sample from a normal distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, factor, /):
        """Rescale the Cholesky factor of a normal distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def residual_white_rms_tree(self, u):
        raise NotImplementedError

    @abc.abstractmethod
    def residual_white_rms_flat(self, u):
        raise NotImplementedError

    @abc.abstractmethod
    def logpdf_tree(self, u):
        raise NotImplementedError

    @abc.abstractmethod
    def logpdf_flat(self, u):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_mean_and_std(cls, mean, std):
        """Construct a normal distribution from mean and standard deviation."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dirac(cls, mean, *, damp):
        """Construct a normal distribution from a Dirac distribution."""
        raise NotImplementedError


class AbstractConditional(abc.ABC):
    """Interface for implementations of manipulating conditionals."""

    def bayes_rule_tree(self, data, rv, conditional, /, *, solve_triu):
        _, reverted = self.revert(rv, conditional, solve_triu=solve_triu)
        data_flat = conditional.noise.tree_flatten.flatten_tree(data)
        return self.apply_flat(data_flat, reverted)

    def bayes_rule_and_logpdf_tree(self, data, rv, conditional, /, *, solve_triu):
        observed, reverted = self.revert(rv, conditional, solve_triu=solve_triu)
        logpdf = observed.logpdf_tree(data)
        data_flat = observed.tree_flatten.flatten_tree(data)
        updated = self.apply_flat(data_flat, reverted)
        return logpdf, updated

    def bayes_rule_and_residual_white_rms_tree(
        self, data, rv, conditional, /, *, solve_triu
    ):
        observed, reverted = self.revert(rv, conditional, solve_triu=solve_triu)
        mahalanobis = observed.residual_white_rms_tree(data)

        data_flat = observed.tree_flatten.flatten_tree(data)
        updated = self.apply_flat(data_flat, reverted)
        return mahalanobis, updated

    @abc.abstractmethod
    def marginalise(self, rv, conditional, /):
        """Compute a marginal of a random variable and conditional."""
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, conditional, /):
        """Revert a parametrisation of a joint distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_flat(self, x, conditional, /):
        """Apply a conditional to a target."""
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, cond1, cond2, /):
        """Merge two conditionals."""
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, cond: LatentCond, /) -> LatentCond:
        """Apply a preconditioner to a conditional."""
        raise NotImplementedError


class ShapeInfo:
    """Information about shapes of Taylor coefficients (lengths, sizes, etc.)."""

    def __init__(self, tcoeffs: Sequence, /):
        # Ensure everything has shapes and dtypes
        tcoeffs = tree.tree_map(np.asarray, tcoeffs)

        # TODO: assert that the tree is a tree of Taylor coefficients
        #       (which means each leaf has the same tree structure)

        # A flattened representation of the Taylor coefficients
        flat, self.all_unravel = tree.ravel_pytree(tcoeffs)
        self.all_flat = structs.ShapeDtypeStruct(flat.shape, flat.dtype)

        # A flattened representation of each Taylor coefficient
        flat, self.single_unravel = tree.ravel_pytree(tcoeffs[0])
        self.single_flat = structs.ShapeDtypeStruct(flat.shape, flat.dtype)

        # The leaves in the Taylor coefficients.
        # Note how each Taylor coefficient can itself be a PyTree,
        # but the leaves must always be arrays.
        # This specific info is especially important for blockdiagonal models.
        # TODO: this somewhat duplicates the above,
        #       but not enough to prioritise refactoring...
        def is_leaf(ell):
            return tree.tree_structure(ell) == tree.tree_structure(tcoeffs[0])

        leaves, self.treedef = tree.tree_flatten_depth_one(tcoeffs)
        _, self.leaf_unravel = tree.ravel_pytree(leaves[0])

        def create_dummy(s):
            return structs.ShapeDtypeStruct(s.shape, s.dtype)

        self.leaves = tree.tree_map(create_dummy, leaves)

    @property
    def num_derivatives(self):
        """The number of derivatives in the SSM."""
        return len(self.leaves) - 1


class AbstractPriorFactory:
    """Interface for prior constructions."""

    def __init__(self, shape_info) -> None:
        self.shape_info = shape_info

    @abc.abstractmethod
    def identity(self, /):
        """Construct an identity conditional (unit linop, zero noise)."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition_wiener_integrated(self, base_scale: Array | None):
        """Construct the transitions for an integrated Wiener process."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition_exponential(self, vf_linear: Callable, base_scale: Array | None):
        raise NotImplementedError

    @abc.abstractmethod
    def to_derivative(self, i, std) -> LatentCond:
        """Construct an observation model for the i'th derivative."""
        raise NotImplementedError

    @abc.abstractmethod
    def prototype_output_scale_calibrated(self):
        """Prototype the calibrated output scale.

        Note how this may differ from the base-output scale.
        For example, base output scales for dense factorisations
        are vector-valued even though the calibrations are scalar.
        See the Robertson DAE examples for why this is helpful.
        """
        raise NotImplementedError


@structs.dataclass
class FactSsmImpl:
    """Implementation of factorized Markovian state-space models."""

    # Linearization and priors: construct RVs and conditionals

    linearize: AbstractLinearizationFactory
    """An implementation of linearization constructors."""

    prior: AbstractPriorFactory
    """An implementation of constructing prior distributions."""

    # Manipulate RVs and conditionals

    conditional: AbstractConditional
    """An implementation of manipulating conditionals."""

    @classmethod
    def from_tcoeffs_dense(cls, tcoeffs_mean, tcoeffs_std, /):
        """Construct a factorised state-space model implementation."""
        shape_info = ShapeInfo(tcoeffs_mean)
        marginal = DenseNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        prior = DensePriorFactory(shape_info=shape_info)

        linearize = DenseLinearizationFactory()
        conditional = DenseConditional()
        ssm = cls(linearize=linearize, conditional=conditional, prior=prior)
        return marginal, ssm

    @classmethod
    def from_tcoeffs_isotropic(cls, tcoeffs_mean, tcoeffs_std, /):
        """Construct a factorised state-space model implementation."""
        shape_info = ShapeInfo(tcoeffs_mean)
        marginal = IsotropicNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        prior = IsotropicPriorFactory(shape_info=shape_info)
        linearize = IsotropicLinearizationFactory()
        conditional = IsotropicConditional()
        ssm = cls(linearize=linearize, prior=prior, conditional=conditional)
        return marginal, ssm

    @classmethod
    def from_tcoeffs_blockdiag(cls, tcoeffs_mean, tcoeffs_std, /):
        """Construct a factorised state-space model implementation."""
        shape_info = ShapeInfo(tcoeffs_mean)
        marginal = BlockDiagNormal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)

        prior = BlockDiagPriorFactory(shape_info=shape_info)
        linearize = BlockDiagLinearizationFactory()
        conditional = BlockDiagConditional()
        ssm = cls(linearize=linearize, prior=prior, conditional=conditional)
        return marginal, ssm


class BlockDiagPriorFactory(AbstractPriorFactory):
    """Implementation of block-diagonal prior constructors."""

    def identity(self, /) -> LatentCond:
        ndim = self.shape_info.num_derivatives + 1
        (d,) = self.shape_info.single_flat.shape
        m0 = np.zeros((d, ndim))
        c0 = np.zeros((d, ndim, ndim))
        tree_flatten = BlockDiagTreeFlatten.from_shape_info(self.shape_info)
        noise = BlockDiagNormal(m0, c0, tree_flatten)
        matrix = np.ones((d, 1, 1)) * np.eye(ndim, ndim)[None, ...]
        return LatentCond.from_linop_and_noise(matrix, noise)

    def transition_wiener_integrated(self, base_scale: Array | None):

        coeff_like = np.ones_like(self.shape_info.single_flat)
        base_scale_expected = self.shape_info.single_unravel(coeff_like)
        if base_scale is None:
            base_scale = base_scale_expected
        else:
            if base_scale.shape != base_scale_expected.shape:
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {base_scale_expected.shape}."
                msg += f" Received: {base_scale.shape}."
                raise ValueError(msg)

        base_scale, _ = tree.ravel_pytree(base_scale)

        num_derivatives = self.shape_info.num_derivatives
        a, q_sqrtm = system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1,))
        precon_fun = preconditioner_taylor(num_derivatives=num_derivatives)

        def discretise(dt, output_scale: Array | None = None):
            p, p_inv = precon_fun(dt)
            if output_scale is None:
                output_scale = np.ones_like(base_scale)
            else:
                output_scale = np.asarray(output_scale)

                if output_scale.shape != base_scale.shape:
                    msg = "The output-scale has the wrong shape."
                    msg += f" Expected: {base_scale.shape}."
                    msg += f" Received: {output_scale.shape}."
                    raise ValueError(msg)
                output_scale, _ = tree.ravel_pytree(output_scale)

            scale = output_scale * base_scale
            # Flatten the scale into something compatible with the flattened SSM
            (d,) = scale.shape

            A_batch = np.ones((d, 1, 1)) * a[None, :, :]
            mean = np.ones((d, 1)) * q0[None, :]
            cholesky = scale[:, None, None] * q_sqrtm[None, :, :]
            tree_flatten = BlockDiagTreeFlatten.from_shape_info(self.shape_info)
            noise = BlockDiagNormal(mean, cholesky, tree_flatten)
            p = np.ones((d, 1)) * p[None, :]
            p_inv = np.ones((d, 1)) * p_inv[None, :]
            return LatentCond(A_batch, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def transition_exponential(self, vf_linear, base_scale):
        del vf_linear
        del base_scale
        msg = "Isotropic IOUPs have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def to_derivative(self, i, std):

        def select(a):
            return np.asarray([a[i]])

        (d,) = self.shape_info.single_flat.shape
        n = self.shape_info.num_derivatives + 1
        x = np.zeros((d, n))
        linop = func.vmap(func.jacrev(select))(x)

        tree_flatten = BlockDiagTreeFlatten.from_shape_info(self.shape_info)
        u_like = tree_flatten.unflatten_array(x)[0]
        noise = BlockDiagNormal.from_mean_and_std([u_like], [std])
        return LatentCond.from_linop_and_noise(linop, noise)

    def prototype_output_scale_calibrated(self):
        # TODO: technically, these should be pytrees according
        # to the leaf structure, right?
        return np.ones(self.shape_info.single_flat.shape)


class IsotropicPriorFactory(AbstractPriorFactory):
    """Implementation of isotropic prior constructors."""

    def identity(self, /) -> LatentCond:
        num = self.shape_info.num_derivatives + 1
        (d,) = self.shape_info.single_flat.shape
        m0 = np.zeros((num, d))
        c0 = np.zeros((num, num))
        tree_flatten = IsotropicTreeFlatten.from_shape_info(self.shape_info)
        noise = IsotropicNormal(m0, c0, tree_flatten)
        matrix = np.eye(num)
        return LatentCond.from_linop_and_noise(matrix, noise)

    def transition_wiener_integrated(self, base_scale: Array | None):

        if base_scale is None:
            base_scale = np.ones(())
        else:
            base_scale = np.asarray(base_scale)
            if base_scale.shape != ():
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {()}."
                msg += f" Received: {base_scale.shape}."
                raise ValueError(msg)

        num_derivatives = self.shape_info.num_derivatives
        (d,) = self.shape_info.single_flat.shape
        A, q_sqrtm = system_matrices_1d_iwp(num_derivatives)
        q0 = np.zeros((num_derivatives + 1, d))
        precon_fun = preconditioner_taylor(num_derivatives=num_derivatives)

        def discretise(dt, output_scale: Array = 1.0):
            output_scale = np.asarray(output_scale)
            if output_scale.shape != ():
                msg = "The base-scale has the wrong shape."
                msg += f" Expected: {()}."
                msg += f" Received: {output_scale.shape}."
                raise ValueError(msg)

            scale = base_scale * output_scale

            p, p_inv = precon_fun(dt)
            tree_flatten = IsotropicTreeFlatten.from_shape_info(self.shape_info)
            noise = IsotropicNormal(q0, scale * q_sqrtm, tree_flatten)

            return LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def transition_exponential(self, vf_linear, base_scale):
        del vf_linear
        del base_scale
        msg = "Isotropic IOUPs have not been implemented (yet.)."
        msg += " If you need them, reach out."
        raise NotImplementedError(msg)

    def to_derivative(self, i, std):

        m = np.zeros((self.shape_info.num_derivatives + 1,))
        linop = func.jacfwd(lambda s: np.asarray([s[i]]))(m)

        u_like = tree.tree_unflatten(self.shape_info.treedef, m)[0]

        # Wrap u_like and std into a list because the random variable
        # expects TaylorCoefficients.
        noise = IsotropicNormal.from_mean_and_std([u_like], [std])
        return LatentCond.from_linop_and_noise(linop, noise)

    def prototype_output_scale_calibrated(self):
        return np.ones(())


class DensePriorFactory(AbstractPriorFactory):
    """Implementation of dense prior constructors."""

    def identity(self, /) -> LatentCond:
        ndim = self.shape_info.num_derivatives + 1
        (d,) = self.shape_info.single_flat.shape
        n = ndim * d

        A = np.eye(n)
        m = np.zeros((n,))
        C = np.zeros((n, n))
        tree_flatten = DenseTreeFlatten.from_shape_info(self.shape_info)
        noise = DenseNormal(m, C, tree_flatten)
        return LatentCond.from_linop_and_noise(A, noise)

    def transition_wiener_integrated(self, base_scale: Array | None):
        Lambda = self._process_base_scale(base_scale)

        # Construct the transitions
        a, q_sqrtm = system_matrices_1d_iwp(self.shape_info.num_derivatives)
        (d,) = self.shape_info.single_flat.shape

        eye_d = np.eye(d)
        A = np.kron(a, eye_d)
        Q = np.kron(q_sqrtm, Lambda)

        q0 = np.zeros(self.shape_info.all_flat.shape)
        precon_fun = preconditioner_taylor(
            num_derivatives=self.shape_info.num_derivatives
        )

        def discretise(dt, output_scale=1.0):
            output_scale = self._process_calibrated_scale(output_scale)

            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            tree_flatten = DenseTreeFlatten.from_shape_info(self.shape_info)
            noise = DenseNormal(q0, output_scale * Q, tree_flatten)
            return LatentCond(A, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def _process_base_scale(self, base_scale):
        # Process the expected shape of the base-scale
        ones_like = np.ones_like(self.shape_info.single_flat)
        base_scale_expected = self.shape_info.single_unravel(ones_like)

        if base_scale is None:
            base_scale = base_scale_expected
        else:
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

    def transition_exponential(self, vf_linear: Array, base_scale: Array | None):
        Lambda = self._process_base_scale(base_scale)

        # Turn the linear vector field into the bottom block of the IOUP
        # by building the matrix-version of the vector field's Jacobian.

        # First, set up a template variable
        leaf_like = np.ones_like(self.shape_info.single_flat)
        leaves = [leaf_like for _ in range(self.shape_info.num_derivatives + 1)]
        tree_flatten = DenseTreeFlatten.from_example(leaves)

        def vf_flat(tcoeffs_flat):
            tcoeffs_tree = tree_flatten.unflatten_array(tcoeffs_flat)
            fx = vf_linear(*tcoeffs_tree)
            return tree.ravel_pytree(fx)[0]

        leaves_flat = tree_flatten.flatten_tree(leaves)
        bottom_block = func.jacfwd(vf_flat)(leaves_flat)

        # Construct the SDE matrices
        (d,) = self.shape_info.single_flat.shape
        num_derivatives = self.shape_info.num_derivatives
        eye_d = np.eye(d)
        a = linalg.diagonal_matrix(np.ones((num_derivatives,)), k=1)
        A = np.kron(a, eye_d)
        A = A.at[-d:, :].set(bottom_block)

        b = np.eye(num_derivatives + 1)[-1][:, None]
        B = np.kron(b, Lambda)

        precon_fun = preconditioner_taylor(num_derivatives=num_derivatives)

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
        q0 = np.zeros(self.shape_info.all_flat.shape)

        def discretise(dt, output_scale=1.0):
            output_scale = self._process_calibrated_scale(output_scale)

            p, p_inv = precon_fun(dt)
            p = np.repeat(p, d)
            p_inv = np.repeat(p_inv, d)

            # Precondition. (I've not seen a big impact, but we do it anyway)
            A_p = dt * p_inv[:, None] * A * p[None, :]
            B_p = np.sqrt(dt) * p_inv[:, None] * B

            eA, L = exp_gram(A_p, B_p)
            tree_flatten = DenseTreeFlatten.from_shape_info(self.shape_info)
            noise = DenseNormal(q0, output_scale * L, tree_flatten)
            return LatentCond(eA, noise, to_latent=p_inv, to_observed=p)

        return discretise

    def to_derivative(self, i, std):
        def select(a):
            return tree.ravel_pytree(self.shape_info.all_unravel(a)[i])[0]

        x = np.zeros(self.shape_info.all_flat.shape)
        linop = func.jacfwd(select)(x)

        data_like = self.shape_info.all_unravel(x)[0]
        noise = DenseNormal.from_mean_and_std([data_like], [std])
        return LatentCond.from_linop_and_noise(linop, noise)

    def prototype_output_scale_calibrated(self):
        return np.ones(())


@structs.dataclass
class DenseTreeFlatten(AbstractTreeFlatten):
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

    @classmethod
    def from_shape_info(cls, shape_info):
        return cls(shape_info.all_unravel)


class DenseNormal(AbstractTreeNormal[DenseTreeFlatten]):
    """Construct a dense implementation of a normal distribution."""

    @classmethod
    def from_dirac(cls, mean, *, damp):
        _verify_taylor_coefficient_pytree(mean)
        std = tree.tree_map(lambda x: np.ones_like(x) * damp, mean)
        return DenseNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        _verify_taylor_coefficient_pytree(mean)
        _verify_taylor_coefficient_pytree(std)

        tree_flatten = DenseTreeFlatten.from_example(mean)

        mean_flat = tree_flatten.flatten_tree(mean)
        std_flat = tree_flatten.flatten_tree(std)

        assert mean_flat.shape == std_flat.shape
        cholesky = linalg.diagonal_matrix(std_flat)
        return cls(mean_flat, cholesky, tree_flatten)

    def mean_tree(self):
        if self.mean_flat.ndim > 1:
            return func.vmap(DenseNormal.mean_tree)(self)
        return self.tree_flatten.unflatten_array(self.mean_flat)

    def std_tree(self):
        if self.mean_flat.ndim > 1:
            return func.vmap(DenseNormal.std_tree)(self)

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


class DenseLinearizationFactory(AbstractLinearizationFactory):
    """Construct a dense linearization factory."""

    def root(self, root, *, jacobian, root_order: int, nlstsq: bool):
        return DenseRoot(root, jacobian=jacobian, root_order=root_order, nlstsq=nlstsq)

    def ode_taylor_0th(self, vf, *, ode_order):
        return DenseOdeTs0(vf, ode_order=ode_order)

    def ode_taylor_1st(self, vf, *, ode_order, jacobian):
        if ode_order > 1:
            raise ValueError

        return DenseOdeTs1(vf, ode_order=ode_order, jacobian=jacobian)


class DenseConditional(AbstractConditional):
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

    def merge(self, cond1: LatentCond, cond2: LatentCond, /) -> LatentCond:
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
        return LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv: DenseNormal, cond: LatentCond, /, *, solve_triu: Callable):
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
        cond_new = LatentCond(
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
        return LatentCond.from_linop_and_noise(A, noise)


class DenseOdeTs0(AbstractOde):
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

        Ms = rv.mean_tree()

        fm = fun(*Ms[: self.ode_order])
        fx = tree.tree_map(lambda s: -s, [fm])
        linop = func.jacrev(a1)(rv.mean_flat)
        noise = DenseNormal.from_dirac(fx, damp=damp)
        cond = LatentCond.from_linop_and_noise(linop, noise)
        return cond, None


class DenseOdeTs1(AbstractOde):
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
        m_tree = rv.mean_tree()

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
        cond = LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class DenseRoot(AbstractRoot):
    """Construct a dense implementation of root-TS1 linearization."""

    def __init__(self, root, *, root_order, jacobian, nlstsq) -> None:
        super().__init__(root, root_order=root_order)
        self.jacobian = jacobian
        self.nlstsq = nlstsq

    def init_linearization(self):
        return self.jacobian.init_jacobian_handler()

    def constraint_flat(self, m: Array, *, t, tree_flatten) -> Array:
        """Evaluate a flattened version of the root constraint."""
        # Unravel the location and extract derivatives
        m_tree = tree_flatten.unflatten_array(m)
        relevant_tcoeffs = m_tree[: self.root_order]

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
        m_tree = rv.mean_tree()
        relevant_tcoeffs = m_tree[: self.root_order]
        root_eval = func.eval_shape(lambda s: [self.root(*s, t=t)], relevant_tcoeffs)

        # Ensure that unravelling does not yield a ShapeDtypeStruct
        root_eval = tree.tree_map(np.zeros_like, root_eval)
        _, unravel = tree.ravel_pytree(root_eval)

        # Turn the linearization into a conditional
        noise = DenseNormal.from_dirac(unravel(fx), damp=damp)

        cond = LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class IsotropicOdeTs0(AbstractOde):
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
        mean = rv.mean_flat
        Ms = rv.mean_tree()
        fx_tree = fun(*(Ms[: self.ode_order]))
        fx = tree.tree_map(lambda s: -s, fx_tree)

        bias = IsotropicNormal.from_dirac([fx], damp=damp)

        linop = func.jacrev(lambda s: s[[self.ode_order], ...])(mean[..., 0])
        cond = LatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class IsotropicOdeTs1(AbstractOde):
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

        mean_tree = rv.mean_tree()
        m0_tree = mean_tree[0]
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
        cond = LatentCond.from_linop_and_noise(linop, noise)
        return cond, state


class BlockDiagOdeTs0(AbstractOde):
    """Construct a block-diagonal implementation of ODE-TS0 linearization."""

    def __init__(self, vf, *, ode_order: int) -> None:
        super().__init__(vf, ode_order=ode_order)

    def init_linearization(self) -> None:
        return None

    def linearize(self, rv, state: None, *, damp: float, t):
        del state
        mean = rv.mean_tree()
        fx = self.vector_field(*(mean[: self.ode_order]), t=t)
        fx = tree.tree_map(lambda s: -s, fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)

        def a1(s):
            return s[[self.ode_order], ...]

        linop = func.vmap(func.jacrev(a1))(rv.mean_flat)

        cond = LatentCond.from_linop_and_noise(linop, bias)
        return cond, None


class BlockDiagOdeTs1(AbstractOde):
    """Construct a block-diagonal implementation of ODE-TS1 linearization."""

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
        mean = rv.mean_flat

        mean_tree = rv.mean_tree()
        m0_tree = mean_tree[0]
        rv0 = BlockDiagNormal.from_dirac([m0_tree], damp=0.0)

        def a1(s):
            return s[[1], ...]

        linop = func.vmap(func.jacrev(a1))(mean)

        def vf_flat(u):
            u_tree = rv0.tree_flatten.unflatten_array(u[:, None])
            fu_tree = fun(*u_tree)
            return rv0.tree_flatten.flatten_tree([fu_tree]).reshape((-1,))

        # Evaluate the linearisation
        m0 = rv.mean_flat[:, 0]
        fx, J_diag, state = self.jacobian.calculate_diagonal(vf_flat, m0, state)

        E1 = func.jacrev(lambda s: s[0])(rv.mean_flat[0])
        linop = linop - J_diag[:, None, None] * E1[None, None, :]

        fx = rv.mean_flat[:, 1] - fx
        fx = fx[..., None]
        diff = func.vmap(lambda a, b: a @ b)(linop, rv.mean_flat)
        fx = fx - diff
        fx = rv0.tree_flatten.unflatten_array(fx)
        bias = BlockDiagNormal.from_dirac([fx], damp=damp)
        cond = LatentCond.from_linop_and_noise(linop, bias)
        return cond, state


class IsotropicLinearizationFactory(AbstractLinearizationFactory):
    """Construct an isotropic linearization-factory."""

    def root(self, root, *, jacobian, root_order: int, nlstsq: Callable | None):
        raise NotImplementedError

    def ode_taylor_1st(self, vf, *, ode_order, jacobian):
        return IsotropicOdeTs1(vf, jacobian=jacobian, ode_order=ode_order)

    def ode_taylor_0th(self, vf, *, ode_order):
        return IsotropicOdeTs0(vf, ode_order=ode_order)


class BlockDiagLinearizationFactory(AbstractLinearizationFactory):
    """Construct a block-diagonal linearization-factory."""

    def root(self, root, *, jacobian, root_order: int, nlstsq: Callable | None):
        raise NotImplementedError

    def ode_taylor_0th(self, vf, *, ode_order):
        return BlockDiagOdeTs0(vf, ode_order=ode_order)

    def ode_taylor_1st(self, vf, *, ode_order, jacobian):
        return BlockDiagOdeTs1(vf, ode_order=ode_order, jacobian=jacobian)


class IsotropicConditional(AbstractConditional):
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
        return LatentCond(
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
        cond_new = LatentCond(
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
        return LatentCond.from_linop_and_noise(A, noise)


class BlockDiagConditional(AbstractConditional):
    """Construct a block-diagonal implementation of manipulating conditionals."""

    def apply_flat(self, x, cond, /):

        def apply_unbatch(s, c):
            s = c.to_latent * s
            m_new = c.to_observed * (c.A @ s + c.noise.mean_flat)
            c_new = c.to_observed[:, None] * c.noise.cholesky_flat
            return BlockDiagNormal(m_new, c_new, cond.noise.tree_flatten)

        return func.vmap(apply_unbatch)(x, cond)

    def marginalise(self, rv, cond, /):
        matrix, noise = cond.A, cond.noise
        assert matrix.ndim == 3
        mean = cond.to_latent * rv.mean_flat
        cholesky = cond.to_latent[:, :, None] * rv.cholesky_flat

        mean_marg = np.einsum("ijk,ik->ij", matrix, mean) + noise.mean_flat

        chol1 = _transpose(matrix @ cholesky)
        chol2 = _transpose(noise.cholesky_flat)
        R_stack = (chol1, chol2)
        cholesky = func.vmap(cholesky_util.sum_of_sqrtm_factors)(R_stack)

        mean_new = cond.to_observed * mean_marg
        cholesky_new = cond.to_observed[:, :, None] * _transpose(cholesky)
        return BlockDiagNormal(mean_new, cholesky_new, cond.noise.tree_flatten)

    def merge(self, cond1, cond2, /):
        # Transform: latent (2) to latent (1)
        T = cond1.to_latent * cond2.to_observed

        # Linear operator
        A1, A2 = cond1.A, T[:, :, None] * cond2.A
        g = func.vmap(lambda a, b: a @ b)(A1, A2)

        # Combined mean
        m1, m2 = T * cond2.noise.mean_flat, cond1.noise.mean_flat
        xi = func.vmap(lambda a, b, c: a @ b + c)(A1, m1, m2)

        # Cholesky factor of combined covariance
        C1, C2 = cond1.noise.cholesky_flat, T[:, :, None] * cond2.noise.cholesky_flat
        R1 = _transpose(func.vmap(lambda a, b: a @ b)(A1, C2))
        R2 = _transpose(C1)
        Xi = func.vmap(cholesky_util.sum_of_sqrtm_factors)((R1, R2))
        Xi = _transpose(Xi)

        # Gather and return
        noise = BlockDiagNormal(xi, Xi, cond1.noise.tree_flatten)
        return LatentCond(
            g, noise, to_latent=cond2.to_latent, to_observed=cond1.to_observed
        )

    def revert(self, rv, cond, /, *, solve_triu):
        # Pull RV into latent space
        mean = cond.to_latent * rv.mean_flat
        cholesky = cond.to_latent[:, :, None] * rv.cholesky_flat

        # QR decomposition
        rv_chol_upper = _transpose(cholesky)
        noise_chol_upper = _transpose(cond.noise.cholesky_flat)
        A_rv_chol_upper = _transpose(cond.A @ cholesky)

        revert_conditional = func.partial(
            cholesky_util.revert_conditional, solve_triu=solve_triu
        )
        revert_vmap = func.vmap(revert_conditional)
        r_obs, (r_cor, gain) = revert_vmap(
            A_rv_chol_upper, rv_chol_upper, noise_chol_upper
        )
        cholesky_obs = np.transpose(r_obs, axes=(0, 2, 1))
        cholesky_cor = np.transpose(r_cor, axes=(0, 2, 1))

        # New backward conditional
        mean_observed = (cond.A @ mean[..., None])[..., 0] + cond.noise.mean_flat
        mean_corrected = mean - (gain @ (mean_observed[..., None]))[..., 0]
        corrected = BlockDiagNormal(mean_corrected, cholesky_cor, rv.tree_flatten)
        bwd = LatentCond(
            gain,
            corrected,
            to_latent=1 / cond.to_observed,
            to_observed=1 / cond.to_latent,
        )

        # Gather observed RV
        mean_observed = cond.to_observed * mean_observed
        cholesky_observed = cond.to_observed[:, :, None] * cholesky_obs
        observed = BlockDiagNormal(
            mean_observed, cholesky_observed, cond.noise.tree_flatten
        )
        return observed, bwd

    def preconditioner_apply(self, cond, /):
        A = cond.to_observed[:, :, None] * cond.A * cond.to_latent[:, None, :]
        mean = cond.to_observed * cond.noise.mean_flat
        cholesky = cond.to_observed[:, :, None] * cond.noise.cholesky_flat
        noise = BlockDiagNormal(mean, cholesky, cond.noise.tree_flatten)
        to_observed = np.ones_like(cond.to_observed)
        to_latent = np.ones_like(cond.to_latent)
        return LatentCond(A, noise, to_observed=to_observed, to_latent=to_latent)


def _transpose(matrix):
    return np.transpose(matrix, axes=(0, 2, 1))


def system_matrices_1d_iwp(num_derivatives):
    """Construct the system matrices of the integrated Wiener process."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...

    # Cholesky factor of flip(hilbert(n))
    Q_1d = cholesky_util.cholesky_hilbert(num_derivatives + 1)
    Q_1d_flipped = np.flip(Q_1d, axis=0)
    Q_1d = linalg.qr_r(Q_1d_flipped.T).T

    scale = np.sign(linalg.diagonal(Q_1d))
    scale = np.where(scale == 0.0, 1.0, scale)
    Q_1d = Q_1d * scale[None, :]
    return A_1d, Q_1d


def preconditioner_taylor(*, num_derivatives):
    """Construct the diagonal preconditioner for Taylor-coefficient state-spaces."""
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)
    powers = powers + 0.5

    def precon(dt):
        dt_abs = np.abs(dt)
        scaling_vector = np.power(dt_abs, powers) / scales
        scaling_vector_inv = np.power(dt_abs, -powers) * scales
        return scaling_vector, scaling_vector_inv

    return precon


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = func.vmap(k, in_axes=(0, None), out_axes=-1)
    return func.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))


@structs.dataclass
class IsotropicTreeFlatten(AbstractTreeFlatten):
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

    @classmethod
    def from_shape_info(cls, shape_info):
        treedef = shape_info.treedef
        unravel_leaf = shape_info.leaf_unravel
        return IsotropicTreeFlatten(treedef, unravel_leaf)


def _verify_taylor_coefficient_pytree(x, /):
    if isinstance(x, Array):
        msg = "Mean must be a pytree, not an array."
        raise TypeError(msg)

    try:
        x_list = [*x]
        shape0 = tree.tree_map(np.shape, x_list[0])
    except Exception as error:
        msg = "Mean must be a pytree of the form [M_1, ..., M_{num_coeffs}], "
        msg += "where each M_i is a pytree of the same structure."
        msg += f" Received: {x}."
        raise ValueError(msg) from error

    for i, x_i in enumerate(x_list):
        xi_shape = tree.tree_map(np.shape, x_i)
        if xi_shape != shape0:
            msg = "All leaves of the mean must have the same shape."
            msg += f" However, leaf {i} has shape {xi_shape}"
            msg += f", while leaf 0 has shape {shape0}"
            raise ValueError(msg)


class IsotropicNormal(AbstractTreeNormal[IsotropicTreeFlatten]):
    """Construct an isotropic normal distribution."""

    @classmethod
    def from_dirac(cls, mean, *, damp):
        _verify_taylor_coefficient_pytree(mean)

        def is_leaf(s):
            return tree.tree_structure(s) == tree.tree_structure(mean[0])

        leaves, structure = tree.tree_flatten_depth_one(mean)
        leaves_ = [np.asarray(damp) for _ in leaves]
        std = tree.tree_unflatten(structure, leaves_)
        return IsotropicNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        _verify_taylor_coefficient_pytree(mean)
        _verify_taylor_coefficient_pytree(std)

        tree_flatten = IsotropicTreeFlatten.from_example(mean)
        loc_flat = tree_flatten.flatten_tree(mean)
        scale_flat = tree_flatten.flatten_tree_scalar(std)

        num_coeffs = len(mean)
        if scale_flat.shape != (num_coeffs,):
            msg = "'std' must have the same pytree structure as mean, "
            msg += "but each leaf must be a scalar instead of an array"
            msg += f"Received: {std}"
            raise ValueError(msg)

        cholesky_flat = linalg.diagonal_matrix(scale_flat)
        return cls(loc_flat, cholesky_flat, tree_flatten)

    def mean_tree(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(IsotropicNormal.mean_tree)(self)
        return self.tree_flatten.unflatten_array(self.mean_flat)

    def std_tree(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(IsotropicNormal.std_tree)(self)
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


@structs.dataclass
class BlockDiagTreeFlatten(AbstractTreeFlatten):
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

    @classmethod
    def from_shape_info(cls, shape_info):
        treedef = shape_info.treedef
        unravel_leaf = shape_info.leaf_unravel
        return BlockDiagTreeFlatten(treedef, unravel_leaf)


class BlockDiagNormal(AbstractTreeNormal[BlockDiagTreeFlatten]):
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
        _verify_taylor_coefficient_pytree(mean)

        std = tree.tree_map(lambda s: damp * np.ones_like(s), mean)
        return BlockDiagNormal.from_mean_and_std(mean, std)

    @classmethod
    def from_mean_and_std(cls, mean, std):
        _verify_taylor_coefficient_pytree(mean)
        _verify_taylor_coefficient_pytree(std)

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

    def mean_tree(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(BlockDiagNormal.mean_tree)(self)

        return self.tree_flatten.unflatten_array(self.mean_flat)

    def std_tree(self):
        if self.mean_flat.ndim > 2:
            return func.vmap(BlockDiagNormal.std_tree)(self)
        diag = np.einsum("ijk,ikj->ij", self.cholesky_flat, self.cholesky_flat)
        std = np.sqrt(diag)
        return self.tree_flatten.unflatten_array(std)

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


DenseNormal.register_pytree_node()
IsotropicNormal.register_pytree_node()
BlockDiagNormal.register_pytree_node()
