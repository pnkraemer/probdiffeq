from probdiffeq.backend import abc, func, inspect, np, tree
from probdiffeq.backend.typing import Array, Callable, Generic, Sequence, TypeVar

__all__ = [
    "AbstractLatentCond",
    "AbstractLinearization",
    "AbstractOde",
    "AbstractResidual",
    "AbstractTreeFlatten",
    "AbstractTreeNormal",
    "FactSsmImpl",
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


class AbstractLatentCond:
    """Conditional distributions in latent space.

    Subclasses implement the SSM-specific operations (marginalise, revert, etc.).
    """

    def __init__(self, A, noise, to_latent, to_observed) -> None:
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

    def bayes_rule_and_residual_white_rms_tree(self, data, rv, /, *, solve_triu):
        """Apply Bayes' rule; also return the whitened residual RMS."""
        observed, reverted = self.revert(rv, solve_triu=solve_triu)
        mahalanobis = observed.residual_white_rms_tree(data)
        data_flat = observed.tree_flatten.flatten_tree(data)
        updated = reverted.apply_flat(data_flat)
        return mahalanobis, updated


AbstractLatentCond._register_as_pytree()


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


class AbstractResidual(AbstractLinearization):
    """Interface for linearizations of general residuals."""

    def __init__(self, residual, /) -> None:
        self.residual = residual

    @property
    def residual_order(self):
        """The order of the residual constraint."""
        return self.residual.num_derivatives_in_args

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(residual={self.residual})"


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

    @abc.abstractmethod
    def identity_conditional(self) -> AbstractLatentCond:
        """Return the identity transition compatible with this distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def prototype_output_scale_calibrated(self):
        """Return a prototype (zero-valued) array for the calibrated output scale."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_derivative(self, i, std) -> "AbstractLatentCond":
        """Construct an observation model that extracts the i-th Taylor coefficient."""
        raise NotImplementedError


def _verify_ioup_signature_and_parse_order(vf) -> int:
    """Parse the vector-field structure from its signature."""
    sig = inspect.signature(vf)
    params = list(sig.parameters.values())

    msg = f"""The dynamics' signature is not compatible with the constraint.

    More precisely, the dynamics are expected to look like

      - f(u, /),
      - f(u, du, /),
      - f(u, du, ddu, /),

    and so on, where the number of positional arguments
    specifies the order of the problem.
    Replace `u`, `du`, and so on with any variable name of your choosing
    but mind the keyword-only argument 't' in the signatures above.

    That said, the arguments

    {[(p.name, p.kind) for p in params]}

    have been detected in the dynamics function.

    Try wrapping the vector field through a pure Python function
    with the correct arguments before passing it to the ODE constraint.

      - No *args or **kwargs
      - No functools.partial

    """

    POSITIONAL = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    KEYWORD = (inspect.Parameter.KEYWORD_ONLY,)

    def is_positional(p):
        return p.kind in POSITIONAL

    def is_keyword(p):
        return p.kind in KEYWORD

    state_args = [p for p in params if is_positional(p)]
    contains_no_positional = len(state_args) == 0
    contains_keyword = len([p for p in params if is_keyword(p)]) > 0

    if contains_no_positional or contains_keyword:
        raise TypeError(msg)

    return len(state_args)


class FactSsmImpl(abc.ABC):
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
    ):
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
    ):
        """Construct a diffuse integrated Wiener process prior."""
        raise NotImplementedError

    def prior_exponential(
        self,
        vf_linear: Callable,
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        """Construct an exponential integrator prior.

        According to https://arxiv.org/abs/2305.14978, but following the numerical
        methods from https://arxiv.org/abs/2310.13462.
        """
        prior_order = _verify_ioup_signature_and_parse_order(vf_linear)
        if prior_order != len(tcoeffs):
            msg = f"""The exponential prior does not match the Taylor coefficients in the SSM.

        Concretely:

        - For two Taylor coefficients, we expect `f(u, du, /)`.
        - For three Taylor coefficients, we expect `f(u, du, ddu, /)`.
        - For four Taylor coefficients, we expect `f(u, du, ddu, dddu, /)`.

        and so on. The passed dynamics correspond to **{prior_order}** Taylor
        coefficients, whereas the state-space model includes **{len(tcoeffs)}**
        Taylor coefficients.
        """
            raise TypeError(msg)
        return self._prior_exponential_impl(
            vf_linear,
            tcoeffs,
            is_exact=is_exact,
            inexact_eps=inexact_eps,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    def _prior_exponential_impl(
        self,
        vf_linear: Callable,
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        raise NotImplementedError

    def prior_exponential_diffuse(
        self,
        vf_linear: Callable,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        """Construct a diffuse exponential integrator prior.

        According to https://arxiv.org/abs/2305.14978, but following the numerical
        methods from https://arxiv.org/abs/2310.13462.
        """
        _verify_ioup_signature_and_parse_order(vf_linear)
        return self._prior_exponential_diffuse_impl(
            vf_linear,
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    # TODO: if we type this with "ODE" instead of Callable,
    # we can avoid this stupid _impl pattern.
    def _prior_exponential_diffuse_impl(
        self,
        vf_linear: Callable,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        raise NotImplementedError

    def prior_ornstein_uhlenbeck_integrated(
        self,
        linop: Callable,
        tcoeffs: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        """Construct an integrated Ornstein-Uhlenbeck prior."""

        def vf_linear(*tc):
            return linop(tc[-1])

        return self._prior_exponential_impl(
            vf_linear,
            tcoeffs,
            is_exact=is_exact,
            inexact_eps=inexact_eps,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    def prior_ornstein_uhlenbeck_integrated_diffuse(
        self,
        linop: Callable,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        """Construct a diffuse integrated Ornstein-Uhlenbeck prior."""

        def vf_linear(*tc):
            return linop(tc[-1])

        return self._prior_exponential_diffuse_impl(
            vf_linear,
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
            output_scale=output_scale,
        )

    # --- Constraint constructors ---

    @abc.abstractmethod
    def constraint_ode_ts0(self, ode, /) -> AbstractOde:
        r"""Create an ODE constraint with zeroth-order Taylor linearisation.

        This constraint handles ODEs of the form

        $$
        \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), ..., t\right)
        $$

        where $k$ is the order of the ODE.

        Related: :class:`probdiffeq.probdiffeq.Constraint`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_ode_ts1(self, ode, /) -> AbstractOde:
        r"""Create an ODE constraint and linearise with a first-order Taylor approximation.

        This constraint handles ODEs of the form

        $$
        \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), ..., t\right)
        $$

        where $k$ is the order of the ODE.

        Related: :class:`probdiffeq.probdiffeq.Constraint`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_residual(self, residual, *, taylor_point=None) -> AbstractResidual:
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
            The strategy to use for finding the linearization point for the Taylor
            expansion. If None, the prior mean is used as the linearization point.

        """
        raise NotImplementedError
