"""Constraint functions.

Examples
--------
>>> from probdiffeq import probdiffeq

Construct ODE constraints as such:

>>> def vf(u, /, *, t):
...     return -u
>>>
>>> ssm = probdiffeq.state_space_model("dense")
>>> constraint = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
>>> print(constraint)
DenseOdeTs1(ode_order=1, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))


Implement high-order ODEs by passing a vector field with additional arguments as such:

>>> def vf(u, du, ddu, /, *, t):
...     return -ddu
>>>
>>> ssm = probdiffeq.state_space_model("isotropic")
>>> constraint = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
>>> print(constraint)
IsotropicOdeTs0(ode_order=3)


Or, use the constraint as a decorator

>>> import functools
>>>
>>> @functools.partial(probdiffeq.constraint_ode_ts0, ssm=ssm)
... def ode(u, du, /, *, t):
...     return -du
>>>
>>> print(ode)
IsotropicOdeTs0(ode_order=2)



"""

from probdiffeq import diffeqjet, ssm_impl
from probdiffeq._probdiffeq import jacobians, linearizations
from probdiffeq.backend import func, inspect, tree
from probdiffeq.backend.typing import Callable, Literal, Protocol, Sequence, TypeVar

__all__ = [
    "Constraint",
    "constraint",
    "constraint_dae",
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


def constraint_ode_ts0(vf: Callable, /, *, ssm: ssm_impl.FactSsmImpl) -> Constraint:
    r"""Create an ODE constraint with zeroth-order Taylor linearisation.

    This constraint handles ODEs of the form

    $$
    \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), \frac{d^2u}{dt^2}(t), ..., t\right)
    $$

    where $k$ is the order of the ODE, which is read off the number of positional arguments in the vector field $f$ (argument `vf`).


    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    Parameters
    ----------
    vf
        The vector field of the ODE. The ODE vector field is assumed to be one of ``f(u, *, t)``, ``f(u, du, *, t)``, etc.
        The order of the ODE is read off the number of positional arguments before t.
        That is, for first-order ODEs, pass ``f(u, *, t)``,
        for second-order ODEs, pass ``f(u, du, *, t)``,
        for third-order ODEs ``f(u, du, ddu, *, t)``, and so on.
    ssm
        The state-space model to use for the constraint.
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    return ssm.linearize.ode_taylor_0th(vf, ode_order=ode_order)


def constraint_ode_ts1(
    vf: Callable,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian: jacobians.JacobianHandler | None = None,
):
    r"""Create an ODE constraint and linearise with a first-order Taylor approximation.

    This constraint handles ODEs of the form

    $$
    \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), \frac{d^2u}{dt^2}(t), ..., t\right)
    $$

    where $k$ is the order of the ODE, which is read off the number of positional arguments in the vector field $f$ (argument `vf`).



    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    Parameters
    ----------
    vf
        The vector field of the ODE. The ODE vector field is assumed to be one of ``f(u, *, t)``, ``f(u, du, *, t)``, etc.
        The order of the ODE is read off the number of positional arguments before t.
        That is, for first-order ODEs, pass ``f(u, *, t)``,
        for second-order ODEs, pass ``f(u, du, *, t)``,
        for third-order ODEs ``f(u, du, ddu, *, t)``, and so on.
    ssm
        The state-space model to use for the constraint.
    jacobian
        The Jacobian handler to use for the linearization.
        If None, a Jacobians are materialized at every stage in dense factorisations
        and Hutchinson-approximated in isotropic or blockdiagonal models.
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if jacobian is None:
        # Use Hutchinson-Jacobian handling for backward compatibility.
        jacobian = jacobians.jacobian_hutchinson_fwd()
    return ssm.linearize.ode_taylor_1st(vf, ode_order=ode_order, jacobian=jacobian)


def _verify_vector_field_signature_and_parse_order(vf: Callable) -> int:
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


def constraint(
    root,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian: jacobians.JacobianHandler | None = None,
    linearization: linearizations.Linearization | None = None,
    jet_order: int | Literal["max"] = "max",
):
    r"""Construct a general constraint.

    This constraint handles problems of the form

    $$
    f\left(u(t), \frac{du}{dt}(t), ..., \frac{d^k u}{dt^k}(t), t\right) = 0
    $$

    where $k$ is the order of the problem, which is read off the number of positional arguments in the root function $f$ (argument `root`).


    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.


    Parameters
    ----------
    root
        The root constraint to linearize. The root constraint is expected to have a signature like ``f(*y, t)`` where the number of positional arguments specifies the order of the problem.
    ssm
        The state-space model to use for the constraint.
    jacobian
        The Jacobian handler to use for the linearization.
        If None, a Jacobians are materialized at every stage in dense factorisations
        and Hutchinson-approximated in isotropic or blockdiagonal models.
    linearization
        The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
        Adjust this variable to use posterior linearization (also known as iterated filtering).
    jet_order
        The order of the jet linearization. If "max", the jet order is as large as possible given the number of Taylor coefficients provided by the solver. Otherwise, the jet order is an integer specifying the order of the jet linearization. For instance, if the solver provides Taylor coefficients
        (What is Jet-linearisation? Stay tuned!).

    """
    root_order = _verify_vector_field_signature_and_parse_order(root)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    if linearization is None:
        linearization = linearizations.linearization_prior_mean()

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
        root_jet, root_order=order, jacobian=jacobian, linearization=linearization
    )


def constraint_dae(
    differential: Callable,
    algebraic: Callable,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian: jacobians.JacobianHandler | None = None,
    linearization: linearizations.Linearization | None = None,
    jet_order_differential: int | Literal["max"] = "max",
    jet_order_algebraic: int | Literal["max"] = "max",
):
    r"""Like `constraint`, but for DAEs.

    The advantage of a dedicated DAE constraint is that algebraic and differential
    roots can enjoy different jet-orders, which increases accuracy.

    This constraint handles problems of the form

    $$
    f\left(u(t), \frac{du}{dt}(t), ..., \frac{d^k u}{dt^k}(t), t\right) = 0,
    ~~~
    g\left(u(t), \frac{du}{dt}(t), ..., \frac{d^{k-1} u}{dt^{k-1}}(t), t\right) = 0
    $$

    where $f$ is the differential part and $g$ the algebraic part of the DAE. The order of the problem is read off the number of positional arguments in the algebraic root $g$ (argument `algebraic`), plus one. For instance, if the algebraic root has two positional arguments, then the problem is a second-order DAE. The differential root (argument `differential`) is expected to have one positional argument more than the algebraic root; in the previous example, the differential root would be expected to have three positional arguments.


    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    Parameters
    ----------
    differential
        The root corresponding to the differential part of the DAE. The root is expected to have a signature like ``f(*y, t)`` where the number of positional arguments specifies the order of the problem.
    algebraic
        The root corresponding to the algebraic part of the DAE. The root is expected to        have a signature like ``f(*y, t)`` where the number of positional arguments specifies the order of the problem.
    ssm
        The state-space model to use for the constraint.
    jacobian
        The Jacobian handler to use for the linearization.
        If None, a Jacobians are materialized at every stage in dense factorisations
        and Hutchinson-approximated in isotropic or blockdiagonal models.
    linearization
        The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
    jet_order_differential
        The order of the jet linearization for the differential root. If "max", the jet order is as large as possible given the number of Taylor coefficients provided by the solver. Otherwise, the jet order is an integer specifying the order of the jet linearization. For instance, if the solver provides Taylor coefficients up to order 5 and the differential root is of order 2, then the jet order can be at most 3.
    jet_order_algebraic
        The order of the jet linearization for the algebraic root. If "max", the jet order is as large as possible given the number of Taylor coefficients provided by the solver. Otherwise, the jet order is an integer specifying the order of the jet linearization. For instance, if the solver provides Taylor coefficients up to order 5 and the algebraic root is of order 1, then the jet order can be at most 4.
    """
    root_order_diff = _verify_vector_field_signature_and_parse_order(differential)
    root_order_alg = _verify_vector_field_signature_and_parse_order(algebraic)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    if linearization is None:
        linearization = linearizations.linearization_prior_mean()

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
            root_jet, root_order="max", jacobian=jacobian, linearization=linearization
        )
    order_diff = root_order_diff + jet_order_differential
    order_alg = root_order_alg + jet_order_algebraic
    order = max(order_diff, order_alg)
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, linearization=linearization
    )
