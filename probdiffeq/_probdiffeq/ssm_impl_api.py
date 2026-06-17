from probdiffeq._probdiffeq import jacobians, problems
from probdiffeq.backend import abc, func, tree
from probdiffeq.backend.typing import (
    TYPE_CHECKING,
    Array,
    Callable,
    Generic,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from probdiffeq._probdiffeq import taylor_points

__all__ = ["StateSpaceModel"]


R = TypeVar("R", bound=Array)
"""A fallback type-variable to cover what isn't covered by the others."""

T = TypeVar("T", bound=Array)
"""A type-variable for Array types.

For example, this variable is used for means and Cholesky factors
in normal distributions.
"""

C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""

# TODO: Consider how we can reuse trace estimators between Jacobians and this one here.
# TODO (cont'd): (do Jacobians implement linear operators? Merge the two classes?)


class AbstractLinOp(abc.ABC):
    def __init__(self, *, n_in, n_out, d_in, d_out):
        self.n_in = n_in
        self.n_out = n_out
        self.d_in = d_in
        self.d_out = d_out

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(n_out={self.n_out}, d_out={self.d_out}"
        msg += f", n_in={self.n_in}, d_in={self.d_in})"
        return msg

    def matmat_dndm(self, M, /):
        matvec = self.matvec_dndm
        matmat = func.vmap(matvec, in_axes=-1, out_axes=-1)
        return matmat(M)

    def matmat_flat(self, M, /):
        matvec = self.matvec_flat
        matmat = func.vmap(matvec, in_axes=-1, out_axes=-1)
        return matmat(M)

    @abc.abstractmethod
    def matvec_ndmd(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def matvec_dndm(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def matvec_flat(self, x, /):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def precon_prototype(self):  # TODO: rename?
        # return ones...
        raise NotImplementedError


class AbstractLatentCond:
    """Conditional distributions in latent space.

    Subclasses implement the SSM-specific operations (marginalise, revert, etc.).
    """

    def __init__(self, A, noise, to_latent, to_observed) -> None:

        if not isinstance(A, AbstractLinOp):
            msg = f"Linear operator expected, but {A} received."
            raise TypeError(msg)

        self.A = A
        self.noise = noise
        self.to_latent = to_latent
        self.to_observed = to_observed

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(A={self.A}, noise={self.noise}"
        msg += f", to_latent={self.to_latent}, to_observed={self.to_observed})"
        return msg

    @classmethod
    def _register_as_pytree(cls) -> None:
        """Register this class (or a subclass) as a JAX pytree."""

        def flatten(cond):
            children = cond.A, cond.noise, cond.to_latent, cond.to_observed
            return children, ()

        def unflatten(_aux, children):
            A, noise, to_latent, to_observed = children
            return cls(A, noise, to_latent, to_observed)

        tree.register_pytree_node(cls, flatten, unflatten)

    @classmethod
    def from_linop_and_noise(cls, A: AbstractLinOp, noise):
        """Construct a latent conditional with unit en- and decoders."""
        if len(noise.batch_shape) > 0:
            return func.vmap(cls.from_linop_and_noise)(A, noise)

        if not isinstance(A, AbstractLinOp):
            msg = f"Linear operator expected, but {A} received."
            raise TypeError(msg)

        to_latent, to_observed = A.precon_prototype
        return cls(A, noise=noise, to_latent=to_latent, to_observed=to_observed)

    def rescale_noise(self, factor, /):
        """Rescale the noise in a conditional."""
        noise = self.noise.rescale_cholesky(factor)
        return self.__class__(
            A=self.A,
            noise=noise,
            to_latent=self.to_latent,
            to_observed=self.to_observed,
        )

    # --- SSM-specific operations (implemented by subclasses) ---

    def marginalise(self, rv, /):
        """Compute the marginal distribution p(y) given rv ~ p(x) and self = p(y|x)."""
        raise NotImplementedError

    def revert(self, rv, /, *, solve_triu):
        """Revert the parametrisation: return (observed, backward_conditional)."""
        raise NotImplementedError

    def apply_flat(self, x, /):
        """Apply this conditional to a flat mean vector x; return a Normal."""
        raise NotImplementedError

    def merge(self, other, /):
        """Merge this conditional with another: compose p(z|y) with p(y|x)."""
        raise NotImplementedError

    def preconditioner_apply(self, /):
        """Apply the preconditioner encoded in to_latent / to_observed."""
        raise NotImplementedError

    # --- Concrete composites built from the abstract ops above ---

    def bayes_rule_tree(self, data, rv, /, *, solve_triu):
        """Apply Bayes' rule and return the posterior (tree-structured data)."""
        _, reverted = self.revert(rv, solve_triu=solve_triu)
        data_flat = self.noise.tree_flatten.flatten_tree(data)
        return reverted.apply_flat(data_flat)

    def bayes_rule_and_logpdf_tree(self, data, rv, /, *, solve_triu):
        """Apply Bayes' rule; also return the log-pdf (tree-structured data)."""
        observed, reverted = self.revert(rv, solve_triu=solve_triu)
        logpdf = observed.logpdf_tree(data)
        data_flat = observed.tree_flatten.flatten_tree(data)
        updated = reverted.apply_flat(data_flat)
        return logpdf, updated

    def bayes_rule_and_residual_whitened_rms_tree(self, data, rv, /, *, solve_triu):
        """Apply Bayes' rule; also return the whitened residual RMS."""
        observed, reverted = self.revert(rv, solve_triu=solve_triu)
        mahalanobis = observed.residual_whitened_rms_tree(data)
        data_flat = observed.tree_flatten.flatten_tree(data)
        updated = reverted.apply_flat(data_flat)
        return mahalanobis, updated


AbstractLatentCond._register_as_pytree()


class AbstractLinearization(abc.ABC):
    """Interface for linearizations."""

    residual_order: int
    """The order of the root-constraint.

    Here, 'order' relates to the highest derivative that the constraint depends on;
    for instance, in first-order ODEs, the residual_order would be two; and in
    second-order ODEs, the residual_order would be three.
    """

    @abc.abstractmethod
    def init_linearization(self):
        """Initialize a linearization."""
        raise NotImplementedError

    @abc.abstractmethod
    def linearize(self, rv, state: None, *, damp: float, t):
        """Evaluate a linearization."""
        raise NotImplementedError


class AbstractResidual(AbstractLinearization):
    """Interface for linearizations of general residuals."""

    def __init__(self, residual, /) -> None:
        self.residual = residual
        self.residual_order = self.residual.num_tcoeffs_in_args

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(residual={self.residual})"


class AbstractOde(AbstractLinearization):
    """Interface for linearizations of ODEs."""

    def __init__(self, *, ode) -> None:
        self.ode = ode
        self.residual_order = self.ode.num_tcoeffs_in_args + 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ode={self.ode})"


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

    @property
    @abc.abstractmethod
    def batch_shape(self):
        """Evaluate the batch-shape."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mean(self):
        """Evaluate the mean."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def std(self):
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
    def residual_whitened_rms_tree(self, u):
        raise NotImplementedError

    @abc.abstractmethod
    def residual_whitened_rms_flat(self, u):
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

    @abc.abstractmethod
    def identity_conditional(self) -> AbstractLatentCond:
        """Return the identity transition compatible with this distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def prototype_output_scale_calibrated(self):
        """Return a prototype (zero-valued) array for the calibrated output scale."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_derivative(self, i, std) -> AbstractLatentCond:
        """Construct an observation model that extracts the i-th Taylor coefficient."""
        raise NotImplementedError


class AbstractPrior(abc.ABC):
    def __init__(self, init, output_scale, /):
        self.init = init
        self.output_scale = output_scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(init={self.init}, output_scale={self.output_scale})"

    @abc.abstractmethod
    def transition(self, *, dt: float, output_scale: Array) -> AbstractLatentCond:
        """Discretize the prior at a time step."""
        raise NotImplementedError


class StateSpaceModel(abc.ABC):
    """Abstract base for factorised Markovian state-space model implementations.

    Construct via `state_space_model_dense`, `state_space_model_blockdiag`,
    or `state_space_model_isotropic`.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    # --- Prior constructors ---

    @abc.abstractmethod
    def prior_wiener_integrated(
        self,
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ) -> AbstractPrior:
        """Construct an integrated Wiener process prior."""
        raise NotImplementedError

    @abc.abstractmethod
    def prior_wiener_integrated_diffuse(
        self,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ) -> AbstractPrior:
        """Construct a diffuse integrated Wiener process prior."""
        raise NotImplementedError

    @abc.abstractmethod
    def prior_exponential(
        self,
        ode: problems.JetOdeAutonomous,
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ) -> AbstractPrior:
        """Construct an exponential integrator prior.

        According to https://arxiv.org/abs/2305.14978, but following the numerical
        methods from https://arxiv.org/abs/2310.13462.
        """
        raise NotImplementedError

    @abc.abstractmethod
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
    ) -> AbstractPrior:
        """Construct a diffuse exponential integrator prior.

        According to https://arxiv.org/abs/2305.14978, but following the numerical
        methods from https://arxiv.org/abs/2310.13462.
        """
        raise NotImplementedError

    def prior_ornstein_uhlenbeck_integrated(
        self,
        linop: Callable[[R], R],
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ) -> AbstractPrior:
        """Construct an integrated Ornstein-Uhlenbeck prior."""

        def autonomous(*, jet_coords):
            return linop(jet_coords[-1])

        ode: problems.JetOdeAutonomous = problems.JetOdeAutonomous(
            autonomous,
            jacobian=jacobians.jacobian_monte_carlo_fwd(),
            num_tcoeffs_in_args=len(tcoeffs),
        )
        return self.prior_exponential(
            ode,
            tcoeffs,
            is_exact=is_exact,
            inexact_eps=inexact_eps,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    def prior_ornstein_uhlenbeck_integrated_diffuse(
        self,
        linop: Callable[[R], R],
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ) -> AbstractPrior:
        """Construct a diffuse integrated Ornstein-Uhlenbeck prior."""

        def autonomous(*, jet_coords):
            return linop(jet_coords[-1])

        ode: problems.JetOdeAutonomous = problems.JetOdeAutonomous(
            autonomous,
            jacobian=jacobians.jacobian_monte_carlo_fwd(),
            num_tcoeffs_in_args=len(tcoeffs_mean),
        )
        return self.prior_exponential_diffuse(
            ode,
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    # --- Linearization constructors ---

    @abc.abstractmethod
    def constraint_ode_ts0(self, ode: problems.JetOde, /) -> AbstractOde:
        r"""Create an ODE constraint with zeroth-order Taylor linearisation.

        This constraint handles ODEs of the form

        $$
        \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), ..., t\right)
        $$

        where $k$ is the order of the ODE.

        Related: :class:`probdiffeq._probdiffeq.ssm_impl_api.AbstractLinearization`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_ode_ts1(self, ode: problems.JetOde, /) -> AbstractOde:
        r"""Create an ODE constraint and linearise with a first-order Taylor approximation.

        This constraint handles ODEs of the form

        $$
        \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), ..., t\right)
        $$

        where $k$ is the order of the ODE.

        Related: :class:`probdiffeq._probdiffeq.ssm_impl_api.AbstractLinearization`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_residual(
        self,
        residual: problems.JetResidual,
        *,
        taylor_point: "taylor_points.TaylorPoint | None" = None,
    ) -> AbstractResidual:
        r"""Construct a general constraint.

        This constraint handles problems of the form

        $$
        f\left(u(t), \frac{du}{dt}(t), ..., t\right) = 0
        $$

        !!! warning "Warning: highly EXPERIMENTAL feature!"
            This function is highly experimental and not safe to use.

        Parameters
        ----------
        residual
            The residual to apply linearization to.
        taylor_point
            The strategy to use for finding the linearization point. If None,
            the prior mean is used as the linearization point.

        """
        raise NotImplementedError
