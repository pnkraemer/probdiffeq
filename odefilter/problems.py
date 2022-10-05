"""Problem types."""

from typing import Any, Callable, Iterable, NamedTuple, Optional, Union


class FirstOrderODE(NamedTuple):
    r"""Vector field for first-order ordinary differential equations.

    First-order ordinary differential equations (ODEs)

    $$
    \dot u(t) = f(u(t), t, \theta)
    $$

    are the default form of a differential equation.
    Higher-order ODEs transform into first-order ODEs (but from a
    computational perspective it is not advisable.)

    This problem type encodes the information about the
    vector field $f$, its Jacobians, e.g., the Jacobian $Jf$ with respect to $u$,
    and more generally, it communicates to the to solvers that an ODE
    with signature $(u, t, \theta)$ is to be expected.
    """

    f: Callable
    r"""ODE vector field $f=f(u, t, \theta)$."""

    jac: Optional[Callable] = None
    r"""Jacobian of the vector field with respect to $u$, $Jf=(Jf)(u, t, \theta)$."""


class SecondOrderODE(NamedTuple):
    r"""Vector field for second-order ordinary differential equations.

    Second-order ordinary differential equations (ODEs)

    $$
    \ddot u(t) = f(u(t), \dot u(t) t, \theta)
    $$

    occur commonly as equations of motion.

    This problem type encodes the information about the
    vector field $f$, its Jacobians, e.g., the Jacobian $Jf$ with respect to $u$,
    and more generally, it communicates to the to solvers that an ODE
    with signature $(u, \dot u, t, \theta)$ is to be expected.
    """

    f: Callable
    r"""ODE vector field $f=f(u, \dot u, t, \theta)$."""

    jac: Optional[Callable] = None
    r"""Jacobian of the vector field with respect to $u$,
    $Jf=(Jf)(u, \dot u, t, \theta)$."""


class InitialValueProblem(NamedTuple):
    """Initial value problem."""

    ode_function: Union[FirstOrderODE, SecondOrderODE]
    """ODE function."""

    initial_values: Union[Any, Iterable[Any]]
    r"""Initial values.
    If the ODE is a first-order equation, the initial value is an array $u_0$.
    If it is a second-order equation,
    the initial values are a tuple of arrays $(u_0, \dot u_0)$.
    If it is an $n$-th order equation, the initial values are an $n$-tuple
    of arrays $(u_0, \dot u_0, ...)$.
    """

    t0: float
    """Initial time-point."""

    t1: float
    """Terminal time-point. Optional."""

    parameters: Any = ()
    """Parameters of the initial value problem."""
