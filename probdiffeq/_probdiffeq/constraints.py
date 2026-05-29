"""Constraint functions."""

from probdiffeq import diffeqjet, ssm_impl
from probdiffeq._probdiffeq.jacobian_handlers import *
from probdiffeq._probdiffeq.loss_functions import *
from probdiffeq._probdiffeq.markov_processes import *
from probdiffeq.backend import func, inspect, tree
from probdiffeq.backend.typing import Callable, Literal, Protocol, Sequence, TypeVar

__all__ = [
    "Constraint",
    "constraint_jet",
    "constraint_jet_dae",
    "constraint_jet_imex",
    "constraint_ode_ts0",
    "constraint_ode_ts1",
]
C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""

N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""


class Constraint(Protocol):
    """An interface for constraints + linearization in probabilistic solvers.

    Related:
    [`constraint_ode_ts0`](#probdiffeq.probdiffeq.constraint_ode_ts0),
    [`constraint_ode_ts1`](#probdiffeq.probdiffeq.constraint_ode_ts1),
    """

    init_linearization: Callable
    """Initialize the linearization of the constraint."""

    linearize: Callable
    """Linearize the constraint."""

    root_order: int
    """The order of the root-constraint.

    Here, 'order' relates to the highest derivative that the
    constraint depends on; for instance, in first-order ODEs,
    the root_order would be two; and in second-order ODEs,
    the root_order would be three.
    """


# TODO: should we go back to an EK0 and EK1 naming to ensure consistency
#       with papers and other libraries?
#       There is no more statistical linear regression
#       (nor will there ever be) so technicalities regarding *how* we linearize
#       are not relevant anymore.
def constraint_ode_ts0(vf, /, *, ssm):
    """Create an ODE constraint with zeroth-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    return ssm.linearize.ode_taylor_0th(vf, ode_order=ode_order)


def constraint_ode_ts1(
    vf, /, *, ssm: ssm_impl.FactSsmImpl, jacobian: JacobianHandler | None = None
):
    """Create an ODE constraint with first-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    The ODE vector field is assumed to be one of ``f(u, *, t)``, ``f(u, du, *, t)``, etc.
    The order of the ODE is read off the number of positional arguments before t.
    That is, for first-order ODEs, pass ``f(u, *, t)``,
    for second-order ODEs, pass ``f(u, du, *, t)``,
    for third-order ODEs ``f(u, du, ddu, *, t)``, and so on.

    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if jacobian is None:
        # Use Hutchinson-Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.ode_taylor_1st(vf, ode_order=ode_order, jacobian=jacobian)


def _verify_vector_field_signature_and_parse_order(vf) -> int:
    """Parse the vector-field structure from its signature."""
    sig = inspect.signature(vf)
    params = list(sig.parameters.values())

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

    msg = f"""The dynamics' signature is not compatible with the constraint.

    More precisely, the dynamics are expected to look like

      - f(u, /, *, t),
      - f(u, du, /, *, t),
      - f(u, du, ddu, /, *, t),

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

    contains_no_positional = len(state_args) == 0
    t_is_not_keyword = not any(is_keyword(p) and p.name == "t" for p in params)
    contains_keyword_other_than_t = any(is_keyword(p) and p.name != "t" for p in params)

    if contains_no_positional or t_is_not_keyword or contains_keyword_other_than_t:
        raise TypeError(msg)

    return len(state_args)


def constraint_jet(
    root,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order: int | Literal["max"] = "max",
):
    """Construct a constraint that implements Jet-linearization.

    (What is Jet-linearisation? Stay tuned!).

    To use posterior linearisation, pass a `nlstsq` implementation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.


    """
    root_order = _verify_vector_field_signature_and_parse_order(root)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        _, unravel_one = tree.ravel_pytree(tcoeffs_all[0])

        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)

            order = root_order + jet_order
            tcoeffs = tcoeffs_all[:order]

        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel_one(s) for s in y]
            fx = root(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

        ps, ss = diffeqjet.jet_unpack_series(flat, root_order)

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)

            # Return a sequence to be compatible with Taylor-coeff logic,
            # but don't bother unflattening the content
            # because the result will be compared to zero anyway
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = "max" if jet_order == "max" else jet_order + root_order
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )


def constraint_jet_imex(
    *,
    implicit: Callable,
    explicit: Callable,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order_implicit="max",
    jet_order_explicit="max",
):
    """Like `constraint_jet`, but for roots summing implicit and explicit terms.

    The advantage of a dedicated IMEX constraint is that gradients can be stopped
    through the explicit part, which enables state-space model factorisation.
    In other words, think of the Jet-IMEX constraint as a generalisation
    of zeroth-order methods to implicit differential equations.


    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """
    root_order_im = _verify_vector_field_signature_and_parse_order(implicit)
    root_order_ex = _verify_vector_field_signature_and_parse_order(explicit)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        _, unravel = tree.ravel_pytree(tcoeffs_all[0])
        fx_implicit = jet_call(
            implicit,
            tcoeffs_all,
            root_order=root_order_im,
            jet_order=jet_order_implicit,
            unravel=unravel,
            t=t,
        )
        fx_explicit = jet_call(
            explicit,
            tcoeffs_all,
            root_order=root_order_ex,
            jet_order=jet_order_explicit,
            unravel=unravel,
            t=t,
        )

        # The Jacobian of the explicit term is ignored,
        # which turns first-order linearisation of root_jet into
        # first-order linearisation of the implicit term but zeroth-order
        # linearisation in the explicit term!
        fx_explicit = [func.stop_gradient(f) for f in fx_explicit]

        # Return the sum: c(x) = Imp(x) + Exp(x)
        return tree.tree_map(lambda a, b: a + b, fx_implicit, fx_explicit)

    def jet_call(fun, tcoeffs_all, /, *, root_order, jet_order, unravel, t):
        """Evaluate the jet'ed root function."""
        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)
            order = jet_order + root_order
            tcoeffs = tcoeffs_all[:order]

        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel(s) for s in y]
            fx = fun(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        coeffs_flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]
        ps, ss = diffeqjet.jet_unpack_series(coeffs_flat, root_order)
        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)
            return [fx]

        primals1, series1 = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals1, *series1]

    if jet_order_explicit == "max" or jet_order_implicit == "max":
        order = "max"
    else:
        order_ex = root_order_ex + jet_order_explicit
        order_im = root_order_im + jet_order_implicit
        order = max(order_ex, order_im)
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )


def constraint_jet_dae(
    differential,
    algebraic,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order_differential: int | Literal["max"] = "max",
    jet_order_algebraic: int | Literal["max"] = "max",
):
    """Like `constraint_jet`, but for DAEs.

    The advantage of a dedicated DAE constraint is that algebraic and differential
    roots can enjoy different jet-orders, which increases accuracy.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """
    root_order_diff = _verify_vector_field_signature_and_parse_order(differential)
    root_order_alg = _verify_vector_field_signature_and_parse_order(algebraic)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        unravel = tree.ravel_pytree(tcoeffs_all[0])[1]

        fx1 = jet_evaluate(
            differential,
            tcoeffs_all,
            jet_order=jet_order_differential,
            root_order=root_order_diff,
            unravel=unravel,
            t=t,
        )
        fx2 = jet_evaluate(
            algebraic,
            tcoeffs_all,
            jet_order=jet_order_algebraic,
            root_order=root_order_alg,
            unravel=unravel,
            t=t,
        )

        return [fx1, fx2]

    def jet_evaluate(fun, tcoeffs_all, /, *, jet_order, root_order, unravel, t):
        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel(s) for s in y]
            fx = fun(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)

            order = jet_order + root_order
            tcoeffs = tcoeffs_all[:order]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]
        ps, ss = diffeqjet.jet_unpack_series(flat, root_order)

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)
            return [fx]
        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    if jet_order_differential == "max" or jet_order_algebraic == "max":
        return ssm.linearize.root(
            root_jet, root_order="max", jacobian=jacobian, nlstsq=nlstsq
        )
    order_diff = root_order_diff + jet_order_differential
    order_alg = root_order_alg + jet_order_algebraic
    order = max(order_diff, order_alg)
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )
