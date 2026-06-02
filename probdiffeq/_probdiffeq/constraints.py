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

from probdiffeq import ssm_impl
from probdiffeq._probdiffeq import linearizations, vector_fields
from probdiffeq.backend.typing import Callable, Protocol, Sequence, TypeVar

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


def constraint_ode_ts0(
    vf: vector_fields.JetFunction, /, *, ssm: ssm_impl.FactSsmImpl
) -> Constraint:
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
    if not isinstance(vf, vector_fields.JetFunction):
        raise TypeError(vf)
    return ssm.linearize.ode_taylor_0th(vector_field=vf)


def constraint_ode_ts1(vf: vector_fields.JetFunction, /, *, ssm: ssm_impl.FactSsmImpl):
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
    if not isinstance(vf, vector_fields.JetFunction):
        raise TypeError(vf)
    return ssm.linearize.ode_taylor_1st(vector_field=vf)


def constraint(
    root,
    *,
    ssm: ssm_impl.FactSsmImpl,
    linearization: linearizations.Linearization | None = None,
    # jet_order: int | Literal["max"] = "max",
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
    linearization
        The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
        Adjust this variable to use posterior linearization (also known as iterated filtering).

    """
    if linearization is None:
        linearization = linearizations.linearization_prior_mean()
    return ssm.linearize.root(root, linearization=linearization)


def constraint_dae(
    dae: vector_fields.DAESystem,
    *,
    ssm: ssm_impl.FactSsmImpl,
    linearization: linearizations.Linearization | None = None,
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
    dae
        The differential-algebraic equation.
    ssm
        The state-space model to use for the constraint.
    linearization
        The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
    """
    if linearization is None:
        linearization = linearizations.linearization_prior_mean()

    return ssm.linearize.dae_posterior_linearization(
        dae=dae, linearization=linearization
    )
