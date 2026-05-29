"""Prior processes."""

from probdiffeq import ssm_impl
from probdiffeq.backend import inspect
from probdiffeq.backend.typing import Array, Callable, Sequence, TypeVar

__all__ = [
    "prior_exponential",
    "prior_exponential_diffuse",
    "prior_ornstein_uhlenbeck_integrated",
    "prior_ornstein_uhlenbeck_integrated_diffuse",
    "prior_wiener_integrated",
    "prior_wiener_integrated_diffuse",
    "state_space_model",
]
C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""


def state_space_model(ssm_fact="dense"):
    """Construct an implementation of a factorised state-space model."""
    if ssm_fact == "dense":
        return ssm_impl.FactSsmImpl.from_dense()

    if ssm_fact == "blockdiag":
        return ssm_impl.FactSsmImpl.from_blockdiag()

    if ssm_fact == "isotropic":
        return ssm_impl.FactSsmImpl.from_isotropic()
    raise ValueError(ssm_fact)


def prior_wiener_integrated(
    tcoeffs: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    # Which of the Taylor coefficients are exact
    is_exact: C | bool = True,
    inexact_eps: float = 1e-6,  # a small value
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct an integrated Wiener process prior."""
    return ssm.prior.wiener_integrated(
        tcoeffs,
        is_exact=is_exact,
        inexact_eps=inexact_eps,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        base_scale=output_scale,
    )


def prior_wiener_integrated_diffuse(
    tcoeffs_mean: C,
    tcoeffs_std: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct a diffuse integrated Wiener process prior."""
    return ssm.prior.wiener_integrated_diffuse(
        tcoeffs_mean,
        tcoeffs_std,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        base_scale=output_scale,
    )


def prior_ornstein_uhlenbeck_integrated(
    linop: Callable,
    tcoeffs: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,  # Which of the Taylor coefficients are exact
    is_exact: C | bool = True,
    inexact_eps: float = 1e-6,  # a small value
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct an integrated Ornstein-Uhlenbeck prior."""

    def vf_linear(*tcoeffs):
        return linop(tcoeffs[-1])

    return ssm.prior.exponential(
        tcoeffs,
        is_exact=is_exact,
        inexact_eps=inexact_eps,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        vf_linear=vf_linear,
        base_scale=output_scale,
    )


def prior_ornstein_uhlenbeck_integrated_diffuse(
    linop: Callable,
    tcoeffs_mean: C,
    tcoeffs_std: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct an integrated Ornstein-Uhlenbeck prior."""

    def vf_linear(*tcoeffs):
        return linop(tcoeffs[-1])

    return ssm.prior.exponential_diffuse(
        tcoeffs_mean,
        tcoeffs_std,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        vf_linear=vf_linear,
        base_scale=output_scale,
    )


def prior_exponential(
    vf_linear: Callable,
    tcoeffs: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    is_exact: C | bool = True,
    inexact_eps: float = 1e-6,  # a small value
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct an exponential integrator prior.

    According to https://arxiv.org/abs/2305.14978, but following the numerical
    methods from https://arxiv.org/abs/2310.13462.
    """
    # TODO: offer a "jacobian" option to enable isotropic and blockdiag implementations?
    prior_order = _verify_ioup_signature_and_parse_order(vf_linear)
    if prior_order != len(tcoeffs):
        msg = f"""The exponential prior does not match the Taylor coefficients in the SSM.

        Concretely:

        - For two Taylor coefficients, we expect `f(u, du, /)`.
        - For three Taylor coefficients, we expect `f(u, du, ddu, /)`.
        - For two Taylor coefficients, we expect `f(u, du, ddu, dddu, /)`.

        and so on. The passed dynamics correspond to **{prior_order}** Taylor
        coefficients, whereas the state-space model includes **{len(tcoeffs)}**
        Taylor coeffients.
        """
        raise TypeError(msg)

    return ssm.prior.exponential(
        tcoeffs,
        is_exact=is_exact,
        inexact_eps=inexact_eps,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        vf_linear=vf_linear,
        base_scale=output_scale,
    )


def prior_exponential_diffuse(
    vf_linear: Callable,
    tcoeffs_mean: C,
    tcoeffs_std: C,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value,
    output_scale: Array | None = None,
):
    """Construct a diffuse exponential integrator prior.

    According to https://arxiv.org/abs/2305.14978, but following the numerical
    methods from https://arxiv.org/abs/2310.13462.
    """
    # TODO: offer a "jacobian" option to enable isotropic and blockdiag implementations?
    prior_order = _verify_ioup_signature_and_parse_order(vf_linear)
    if prior_order != len(tcoeffs_mean):
        msg = f"""The exponential prior does not match the Taylor coefficients in the SSM.

        Concretely:

        - For two Taylor coefficients, we expect `f(u, du, /)`.
        - For three Taylor coefficients, we expect `f(u, du, ddu, /)`.
        - For two Taylor coefficients, we expect `f(u, du, ddu, dddu, /)`.

        and so on. The passed dynamics correspond to **{prior_order}** Taylor
        coefficients, whereas the state-space model includes **{len(tcoeffs_mean)}**
        Taylor coeffients.
        """
        raise TypeError(msg)

    return ssm.prior.exponential_diffuse(
        tcoeffs_mean,
        tcoeffs_std,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
        vf_linear=vf_linear,
        base_scale=output_scale,
    )


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
