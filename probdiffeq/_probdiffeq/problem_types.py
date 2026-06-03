"""Vector field API.

An interface for giving all vector fields a common API internally, while offering
flexibility on the outside. Use-cases:

- Handle higher-order problems without reducing them to first-order form, which increases the state dimensionality
and slows down the simulation

- Handle ODEs and DAEs with the same backend code

- Type the constraints and other solver components.


Examples
--------
>>> import inspect
>>>
>>> def f(y, *, t):
...     return y
>>>
>>> print(ode_function(f))
ODE(num_derivatives_in_args=1, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))

Among other things, the vector field wrappers ensure that all internal representations
of the ODEs have the same signature, which means that sometimes (eg for first-order problems),
the internal representation does not match the user-specified one.

>>> print(inspect.signature(f))
(y, *, t)
>>> print(inspect.signature(ode_function(f)))
(*jet_coords: *tuple[~T], t: jax.Array) -> ~T

This API difference is more pronounced for higher-order problems:

>>> def f_second_order(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(ode_function_second_order(f_second_order))
ODE(num_derivatives_in_args=2, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))
>>>
>>> print(inspect.signature(f_second_order))
(y, dy, /, *, t)
>>> print(inspect.signature(ode_function_second_order(f_second_order)))
(*jet_coords: *tuple[~T], t: jax.Array) -> ~T

"""

from probdiffeq._probdiffeq import jacobians, utilities
from probdiffeq.backend import func, tree
from probdiffeq.backend.typing import Any, Array, Generic, Protocol, Sequence, TypeVar

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)

__all__ = [
    "ODE",
    "AbstractJetFunction",
    "FirstOrderODEProtocol",
    "JetFunction",
    "OneJetFunctionProtocol",
    "SecondOrderODEProtocol",
    "ZeroJetFunctionProtocol",
    "dae_system",
    "jet_lift",
    "ode_function",
    "ode_function_second_order",
    "root_state",
    "root_state_and_velocity",
]


class AbstractJetFunction:
    """A jet function, ie a function that operates on jet coordinates (y, y', ..., t).

    This is typically used to define right-hand sides of (high-order) ODEs
    and residuals in implicit differential equations.

    Specifications include JetFunctions (y, y', ..., t) -> Any, which define DAEs
    and implicit differential equations, as well as ODEs, where the output type
    matches the input types.
    """

    def __init__(
        self,
        jet_function,
        jacobian: jacobians.JacobianHandler,
        num_derivatives_in_args: int,
    ):
        self.jet_function = jet_function

        self.jacobian = jacobian
        self.num_derivatives_in_args = num_derivatives_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives_in_args={self.num_derivatives_in_args}, jacobian={self.jacobian})"


class ODE(AbstractJetFunction, Generic[T]):
    def __call__(self, *jet_coords: *tuple[T], t: Array) -> T:
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.jet_function(jet_coords=jet_coords, t=t)


class FirstOrderODEProtocol(Protocol[T]):
    def __call__(self, u: T, /, *, t: float) -> T: ...


def ode_function(
    func: FirstOrderODEProtocol, /, *, jacobian: jacobians.JacobianHandler | None = None
) -> ODE:
    """Construct a description of an explicit ODE y^{(K)} = f(y, y', ..., y^{(K-1)}, t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    return ODE(jetfunc, jacobian=jacobian, num_derivatives_in_args=1)


class SecondOrderODEProtocol(Protocol[T]):
    def __call__(self, u: T, du: T, /, *, t: float) -> T: ...


def ode_function_second_order(
    func: SecondOrderODEProtocol,
    /,
    *,
    jacobian: jacobians.JacobianHandler | None = None,
) -> ODE:
    """Construct a description of an explicit ODE y^{(K)} = f(y, y', ..., y^{(K-1)}, t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    return ODE(jetfunc, jacobian=jacobian, num_derivatives_in_args=2)


class JetFunction(AbstractJetFunction):
    """A jet function, ie a function that operates on jet coordinates (y, y', ..., t).

    This is typically used to define right-hand sides of (high-order) ODEs
    and residuals in implicit differential equations.
    """

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> Any:
        """Make the vector field callable like the original user function to hide the "sophisticated" API."""
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.jet_function(jet_coords=jet_coords, t=t)


class ZeroJetFunctionProtocol(Protocol[T_contra]):
    def __call__(self, u: T_contra, /, *, t: float) -> Any: ...


def root_state(
    func: ZeroJetFunctionProtocol,
    /,
    *,
    jacobian: jacobians.JacobianHandler | None = None,
) -> JetFunction:
    """Construct a description of a root f(u, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    return JetFunction(jetfunc, jacobian=jacobian, num_derivatives_in_args=1)


class OneJetFunctionProtocol(Protocol[T_contra]):
    def __call__(self, u: T_contra, du: T_contra, /, *, t: float) -> Any: ...


def root_state_and_velocity(
    func: OneJetFunctionProtocol,
    /,
    *,
    jacobian: jacobians.JacobianHandler | None = None,
) -> JetFunction:
    """Construct a description of a root f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_hutchinson_fwd()

    return JetFunction(jetfunc, jacobian=jacobian, num_derivatives_in_args=2)


class dae_system:
    """Differential-algebraic equations."""

    def __init__(self, differential: JetFunction, algebraic: JetFunction):
        self.differential = differential
        self.algebraic = algebraic

    def __repr__(self):
        return f"{self.__class__.__name__}(differential={self.differential}, algebraic={self.algebraic})"


F = TypeVar("F", bound=AbstractJetFunction)


def jet_lift(jetfun: F, lift_by: int) -> F:
    """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
    if not isinstance(lift_by, int):
        raise TypeError

    def jetfun_lifted(*, jet_coords: Sequence[T], t) -> Sequence[T]:
        tcoeffs_all = jet_coords
        _, unravel_one = tree.ravel_pytree(tcoeffs_all[0])

        lift_by_upper = len(tcoeffs_all) - jetfun.num_derivatives_in_args
        if lift_by < 0 or lift_by > lift_by_upper:
            msg = "The provided jet-order is incompatible with the root order."
            msg += f" Expected: 0 <= lift_by <= {lift_by_upper}."
            msg += f" Received: lift_by == {lift_by}."
            raise ValueError(msg)
        order = jetfun.num_derivatives_in_args + lift_by
        tcoeffs = tcoeffs_all[:order]

        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel_one(s) for s in y]
            fx = jetfun.jet_function(jet_coords=y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

        ps, ss = utilities.jet_coords_to_primals_and_series(
            flat, jetfun.num_derivatives_in_args
        )

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)

            # Return a sequence to be compatible with Taylor-coeff logic,
            # but don't bother unflattening the content
            # because the result will be compared to zero anyway
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = jetfun.num_derivatives_in_args + lift_by
    return type(jetfun)(
        jetfun_lifted, num_derivatives_in_args=order, jacobian=jetfun.jacobian
    )
