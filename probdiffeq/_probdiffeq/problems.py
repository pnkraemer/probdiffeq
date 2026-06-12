"""Vector field API.

An interface for giving all vector fields a common API internally, while offering
flexibility on the outside. Use-cases:

- Handle higher-order problems without reducing them to first-order form, which increases the state dimensionality
and slows down the simulation

- Handle ODEs and DAEs with the same backend code

- Type the constraints and other solver components.


Examples
--------
>>> @ode
... def f(y, *, t):
...     return y
>>>
>>> print(f)
ODEFunction(num_tcoeffs_in_args=1, jacobian=jacobian_monte_carlo_fwd(seed=1, num_probes=10))


Higher-order problems:

>>> @ode_order_two
... def f(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(f)
ODEFunction(num_tcoeffs_in_args=2, jacobian=jacobian_monte_carlo_fwd(seed=1, num_probes=10))

General constraints:

>>> import jax.numpy as jnp
>>>
>>> @residual_position
... def g(y, /, *, t):
...     return jnp.abs2(y)
>>>
>>> print(g)
Residual(num_tcoeffs_in_args=1, jacobian=jacobian_monte_carlo_fwd(seed=1, num_probes=10))

Higher-order constraints:

>>> @residual_velocity
... def g(y, dy, /, *, t):
...     return jnp.abs2(dy)
>>>
>>> print(g)
Residual(num_tcoeffs_in_args=2, jacobian=jacobian_monte_carlo_fwd(seed=1, num_probes=10))



"""

from probdiffeq._probdiffeq import jacobians, utilities
from probdiffeq.backend import func, tree
from probdiffeq.backend.typing import Any, Array, Generic, Protocol, Sequence, TypeVar

__all__ = [
    "ODEFunction",
    "ODEFunctionAutonomous",
    "ProtocolODEAutonomous",
    "ProtocolODEAutonomousOrderTwo",
    "ProtocolODEFirstOrder",
    "ProtocolODEOrderTwo",
    "ProtocolResidualAcceleration",
    "ProtocolResidualPosition",
    "ProtocolResidualVelocity",
    "Residual",
    "ode",
    "ode_autonomous",
    "ode_autonomous_order_arbitrary",
    "ode_autonomous_order_two",
    "ode_jet_lift",
    "ode_order_arbitrary",
    "ode_order_two",
    "residual_acceleration",
    "residual_from_stack",
    "residual_jet_lift",
    "residual_position",
    "residual_velocity",
]


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class _AbstractJetFunction:
    """A jet function, ie a function that operates on jet coordinates (y, y', ..., t).

    This is typically used to define right-hand sides of (high-order) ODEs
    and residuals in implicit differential equations.

    Specifications include JetFunctions (y, y', ..., t) -> Any, which define DAEs
    and implicit differential equations, as well as ODEs, where the output type
    matches the input types.
    """

    def __init__(self, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int):
        self.jacobian = jacobian
        self.num_tcoeffs_in_args = num_tcoeffs_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_tcoeffs_in_args={self.num_tcoeffs_in_args}, jacobian={self.jacobian})"


class ProtocolODEFirstOrder(Protocol[T]):
    def __call__(self, u: T, /, *, t: float) -> T: ...


class ProtocolODEOrderTwo(Protocol[T]):
    def __call__(self, u: T, du: T, /, *, t: float) -> T: ...


class ODEFunction(_AbstractJetFunction, Generic[T]):
    def __init__(
        self, vector_field, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int
    ):
        super().__init__(jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args)
        self.vector_field = vector_field

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> T:
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.vector_field(jet_coords=jet_coords, t=t)


class ODEFunctionAutonomous(ODEFunction[T]):
    """An autonomous ODE y^(k) = f(y, y', ...) where f does not depend on t."""

    def __init__(
        self, autonomous, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int
    ):
        def vector_field(*, jet_coords, t):
            del t
            return autonomous(jet_coords=jet_coords)

        super().__init__(
            vector_field, jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args
        )
        self.autonomous = autonomous


def ode(func: ProtocolODEFirstOrder, /, *, jacobian: jacobians.Jacobian | None = None):
    """Construct a description of an  ODE y' = f(y, t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunction(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=1)


def ode_order_two(
    func: ProtocolODEOrderTwo, /, *, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an  ODE y'' = f(y, y', t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunction(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=2)


# No typing because arbitrary order is difficult to type (unlike ode and ode_order_two)


def ode_order_arbitrary(
    func, /, *, num_tcoeffs_in_args: int, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an ODE of arbitrary order."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        return func(*jet_coords[:num_tcoeffs_in_args], t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunction(
        jetfunc, jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args
    )


class ProtocolODEAutonomous(Protocol[T]):
    def __call__(self, u: T, /) -> T: ...


def ode_autonomous(
    func: ProtocolODEAutonomous, /, *, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an autonomous ODE y' = f(y)."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        (y,) = jet_coords
        return func(y)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunctionAutonomous(autonomous, jacobian=jacobian, num_tcoeffs_in_args=1)


class ProtocolODEAutonomousOrderTwo(Protocol[T]):
    def __call__(self, u: T, du: T, /) -> T: ...


def ode_autonomous_order_two(
    func: ProtocolODEAutonomousOrderTwo,
    /,
    *,
    jacobian: jacobians.Jacobian | None = None,
):
    """Construct a description of an autonomous ODE y'' = f(y, y')."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        (y, dy) = jet_coords
        return func(y, dy)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunctionAutonomous(autonomous, jacobian=jacobian, num_tcoeffs_in_args=2)


# No typing because arbitrary order is difficult to type (unlike ode and ode_order_two)


def ode_autonomous_order_arbitrary(
    func, /, *, num_tcoeffs_in_args: int, jacobian: jacobians.Jacobian | None = None
) -> "ODEFunctionAutonomous":
    """Construct an autonomous ODE of arbitrary order."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        return func(*jet_coords[:num_tcoeffs_in_args])

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return ODEFunctionAutonomous(
        autonomous, jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args
    )


class Residual(_AbstractJetFunction):
    """A residual on jet coordinates, ie a function that operates on (y, y', ..., t)."""

    def __init__(
        self, residual_function, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int
    ):
        super().__init__(jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args)
        self.residual_function = residual_function

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> Any:
        """Make the vector field callable like the original user function to hide the "sophisticated" API."""
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.residual_function(jet_coords=jet_coords, t=t)


class ProtocolResidualPosition(Protocol[T_contra]):
    def __call__(self, u: T_contra, /, *, t: float) -> Any: ...


def residual_position(
    func: ProtocolResidualPosition, /, *, jacobian: jacobians.Jacobian | None = None
) -> Residual:
    """Construct a description of a residual f(u, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=1)


def residual_from_stack(*residual_stack: *tuple[Residual, ...]) -> Residual:
    """Construct a description of a residual by stacking other residuals."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        return [
            r.residual_function(jet_coords=jet_coords[: r.num_tcoeffs_in_args], t=t)
            for r in residual_stack
        ]

    nums = [r.num_tcoeffs_in_args for r in residual_stack]
    num_args = max(nums)
    jacobian = residual_stack[0].jacobian
    return Residual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=num_args)


class ProtocolResidualVelocity(Protocol[T_contra]):
    def __call__(self, u: T_contra, du: T_contra, /, *, t: float) -> Any: ...


def residual_velocity(
    func: ProtocolResidualVelocity, /, *, jacobian: jacobians.Jacobian | None = None
) -> Residual:
    """Construct a description of a residual f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=2)


class ProtocolResidualAcceleration(Protocol[T_contra]):
    def __call__(
        self, u: T_contra, du: T_contra, ddu: T_contra, /, *, t: float
    ) -> Any: ...


def residual_acceleration(
    func: ProtocolResidualAcceleration, /, *, jacobian: jacobians.Jacobian | None = None
) -> Residual:
    """Construct a description of a residual f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy, ddy) = jet_coords
        return func(y, dy, ddy, t=t)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=3)


def residual_jet_lift(residual: Residual, lift_by: int) -> Residual:
    """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
    if not isinstance(lift_by, int):
        raise TypeError

    def residual_lifted(*, jet_coords: Sequence[T], t) -> Sequence[T]:
        tcoeffs_all = jet_coords
        _, unravel_one = tree.ravel_pytree(tcoeffs_all[0])

        lift_by_upper = len(tcoeffs_all) - residual.num_tcoeffs_in_args
        if lift_by < 0 or lift_by > lift_by_upper:
            msg = "The provided jet-order is incompatible with the residual order."
            msg += f" Expected: 0 <= lift_by <= {lift_by_upper}."
            msg += f" Received: lift_by == {lift_by}."
            raise ValueError(msg)
        order = residual.num_tcoeffs_in_args + lift_by
        tcoeffs = tcoeffs_all[:order]

        # Flatten the residual because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel_one(s) for s in y]
            fx = residual.residual_function(jet_coords=y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

        ps, ss = utilities.jet_coords_to_primals_and_series(
            flat, residual.num_tcoeffs_in_args
        )

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)

            # Return a sequence to be compatible with Taylor-coeff logic,
            # but don't bother unflattening the content
            # because the result will be compared to zero anyway
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = residual.num_tcoeffs_in_args + lift_by
    return Residual(
        residual_lifted, num_tcoeffs_in_args=order, jacobian=residual.jacobian
    )


def ode_jet_lift(ode_fun: ODEFunction, lift_by: int) -> ODEFunction:
    """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
    if not isinstance(lift_by, int):
        raise TypeError

    def residual_lifted(*, jet_coords: Sequence[T], t) -> Sequence[T]:
        tcoeffs_all = jet_coords
        _, unravel_one = tree.ravel_pytree(tcoeffs_all[0])

        lift_by_upper = len(tcoeffs_all) - residual.num_tcoeffs_in_args
        if lift_by < 0 or lift_by > lift_by_upper:
            msg = "The provided jet-order is incompatible with the residual order."
            msg += f" Expected: 0 <= lift_by <= {lift_by_upper}."
            msg += f" Received: lift_by == {lift_by}."
            raise ValueError(msg)
        order = residual.num_tcoeffs_in_args + lift_by
        tcoeffs = tcoeffs_all[:order]

        # Flatten the residual because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel_one(s) for s in y]
            fx = residual.residual_function(jet_coords=y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

        ps, ss = utilities.jet_coords_to_primals_and_series(
            flat, residual.num_tcoeffs_in_args
        )

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)

            # Return a sequence to be compatible with Taylor-coeff logic,
            # but don't bother unflattening the content
            # because the result will be compared to zero anyway
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = ode_fun.num_tcoeffs_in_args + lift_by
    return ODEFunction(
        residual_lifted, num_tcoeffs_in_args=order, jacobian=ode_fun.jacobian
    )
