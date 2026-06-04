from probdiffeq.backend import abc, func, np, tree
from probdiffeq.backend.typing import Array, Callable, Generic, Sequence, TypeVar

__all__ = [
    "AbstractConditional",
    "AbstractLinearization",
    "AbstractLinearizationFactory",
    "AbstractOde",
    "AbstractPriorFactory",
    "AbstractRoot",
    "AbstractTreeFlatten",
    "AbstractTreeNormal",
    "LatentCond",
]


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
    """Interface for linearizations of general residuals."""

    def __init__(self, residual, /) -> None:
        self.residual = residual

    @property
    def residual_order(self):
        """The order of the residual constraint."""
        return self.residual.num_derivatives_in_args

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(residual={self.residual})"


class AbstractDAE(AbstractLinearization):
    """Interface for linearizations of general residuals."""

    def __init__(self, *, dae, linearization) -> None:
        self.dae = dae
        self.linearization = linearization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dae={self.dae}, linearization={self.linearization})"


class AbstractOde(AbstractLinearization):
    """Interface for linearizations of ODEs."""

    def __init__(self, *, ode) -> None:
        self.ode = ode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ode={self.ode})"

    @property
    def residual_order(self):
        """The order of the residual constraint."""
        return self.ode.num_derivatives_in_args + 1


class AbstractLinearizationFactory(abc.ABC):
    """Interface for linearization factories."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def residual(self, residual, *, linearization: Callable | None) -> AbstractRoot:
        """Construct an implementation of 1st-order Taylor-linearization for residuals."""
        raise NotImplementedError

    @abc.abstractmethod
    def dae(self, *, dae, linearization) -> AbstractDAE:
        raise NotImplementedError

    @abc.abstractmethod
    def ode_taylor_0th(self, *, ode) -> AbstractOde:
        """Construct an implementation of 0th-order Taylor-linearization for ODEs."""
        raise NotImplementedError

    @abc.abstractmethod
    def ode_taylor_1st(self, *, ode) -> AbstractOde:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

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


class AbstractPriorFactory(abc.ABC):
    """Interface for prior constructions."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def identity(self, /):
        """Construct an identity conditional (unit linop, zero noise)."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """Construct the transitions for an integrated Wiener process."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """Construct the transitions for an integrated Wiener process."""
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
