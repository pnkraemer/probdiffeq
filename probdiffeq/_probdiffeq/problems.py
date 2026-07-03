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
JetOde(num_tcoeffs_in_args=1, jacobian=jacobian_monte_carlo_rev(seed=1, num_probes=10), tcoeff_indices_output=[1])


Higher-order problems:

>>> @ode_order_two
... def f(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(f)
JetOde(num_tcoeffs_in_args=2, jacobian=jacobian_monte_carlo_rev(seed=1, num_probes=10), tcoeff_indices_output=[2])

General constraints:

>>> import jax.numpy as jnp
>>>
>>> @residual_position
... def g(y, /, *, t):
...     return jnp.abs2(y)
>>>
>>> print(g)
JetResidual(num_tcoeffs_in_args=1, jacobian=jacobian_monte_carlo_rev(seed=1, num_probes=10))

Higher-order constraints:

>>> @residual_velocity
... def g(y, dy, /, *, t):
...     return jnp.abs2(dy)
>>>
>>> print(g)
JetResidual(num_tcoeffs_in_args=2, jacobian=jacobian_monte_carlo_rev(seed=1, num_probes=10))



"""

from probdiffeq._probdiffeq import jacobians, utilities
from probdiffeq.backend import func, np, tree
from probdiffeq.backend.typing import Any, Array, Generic, Protocol, Sequence, TypeVar

__all__ = [
    "JetAbstract",
    "JetOde",
    "JetOdeAutonomous",
    "JetResidual",
    "ode",
    "ode_autonomous",
    "ode_autonomous_order_arbitrary",
    "ode_autonomous_order_two",
    "ode_order_arbitrary",
    "ode_order_two",
    "residual_acceleration",
    "residual_from_ode",
    "residual_from_stack",
    "residual_position",
    "residual_velocity",
]


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class JetAbstract:
    """A jet function, ie a function that operates on jet coordinates (u, u', ..., t).

    This is typically used to define right-hand sides of (high-order) ODEs
    and residuals in implicit differential equations.

    Specifications include JetFunctions (u, u', ..., t) -> Any, which define DAEs
    and implicit differential equations, as well as ODEs, where the output type
    matches the input types.
    """

    def __init__(self, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int):
        self.jacobian = jacobian
        self.num_tcoeffs_in_args = num_tcoeffs_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_tcoeffs_in_args={self.num_tcoeffs_in_args}, jacobian={self.jacobian})"

    def lift(self, fun, /, *, lift_by: int):
        """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""

        def fun_lifted(*, jet_coords: Sequence[T], t) -> Sequence[T]:

            # Check that the lift_by argument is feasible:
            lift_by_upper = len(jet_coords) - self.num_tcoeffs_in_args
            if lift_by < 0 or lift_by > lift_by_upper:
                msg = "The provided jet-order is incompatible with the residual order."
                msg += f" Expected: 0 <= lift_by <= {lift_by_upper}."
                msg += f" Received: lift_by == {lift_by}."
                raise ValueError(msg)

            order = self.num_tcoeffs_in_args + lift_by
            tcoeffs = jet_coords[:order]

            # Learn how to unflatten the outputs. We need this for
            # the output types to be consistent with the original
            # pytree structure (needed especially for ODE jet-lifting)
            [out_like] = func.eval_shape(
                fun, jet_coords=jet_coords[: self.num_tcoeffs_in_args], t=t
            )
            out_like = tree.tree_map(np.zeros_like, out_like)
            _, unravel_outputs = tree.ravel_pytree(out_like)

            # Flatten the residual because jax.experimental.jet is a bit high maintenance :)
            _, unravel_inputs = tree.ravel_pytree(jet_coords[0])
            flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

            def jet_call(*y):
                y_tree = [unravel_inputs(s) for s in y]
                fx = fun(jet_coords=y_tree, t=t)
                return tree.ravel_pytree(fx)[0]

            # Process the Taylor-coefficient list into the primals/series combination
            # needed for jax.experimental.jet
            ps, ss = utilities.jet_coords_to_primals_and_series(
                flat, self.num_tcoeffs_in_args
            )

            # If there are no series, then we can just call the function directly and return the result.
            if len(tree.tree_leaves(ss)) == 0:
                fx = jet_call(*ps)

                # Return a sequence to be compatible with Taylor-coeff logic,
                # but don't bother unflattening the content
                # because the result will be compared to zero anyway
                return [unravel_outputs(fx)]

            # Call the jet, combine the primals and series,
            # and return the result as a sequence of Taylor coefficients
            primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
            coeffs_out = [primals, *series]
            return [unravel_outputs(c) for c in coeffs_out]

        return fun_lifted


class _ProtocolODEFirstOrder(Protocol[T]):
    def __call__(self, u: T, /, *, t: float) -> T: ...


class _ProtocolODEOrderTwo(Protocol[T]):
    def __call__(self, u: T, du: T, /, *, t: float) -> T: ...


class _JetOdeCommon(JetAbstract, Generic[T]):
    def __init__(
        self,
        vector_field,
        jacobian: jacobians.Jacobian,
        num_tcoeffs_in_args: int,
        tcoeff_indices_output: list[int],
    ):
        # TODO: turn the num_tcoeffs_in_args argument into tcoeff_indices_input.
        #       However, this requires manipulating the jet_lift logic to handle
        #       inputs like (2, 4, 5). There is a well-defined solution,
        #       but, for now, spelling out this solution is too complicated for me.
        super().__init__(jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args)
        self.vector_field = vector_field
        self.tcoeff_indices_output = tcoeff_indices_output

    def __repr__(self):
        return f"{self.__class__.__name__}(num_tcoeffs_in_args={self.num_tcoeffs_in_args}, jacobian={self.jacobian}, tcoeff_indices_output={self.tcoeff_indices_output})"

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> T:
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))

        if self.is_jet_extended:
            raise ValueError

        [fx] = self.vector_field(jet_coords=jet_coords, t=t)
        return fx

    @property
    def tcoeff_indices_input(self) -> list[int]:
        return list(range(self.num_tcoeffs_in_args))

    @property
    def is_jet_extended(self):
        """Whether or not the ODE is the result of Jet-extension.

        If true, some functionality is no longer available.
        For example, jet initialisation or stepsize initialisation,
        both of which assume "traditional" vector field signatures.
        """
        return len(self.tcoeff_indices_output) > 1


class JetOde(_JetOdeCommon, Generic[T]):
    def jet_lift_max(self, *, num_tcoeffs: int) -> "JetOde":
        if self.is_jet_extended:
            raise ValueError
        [output_idx] = self.tcoeff_indices_output
        assert output_idx >= self.num_tcoeffs_in_args

        # E.g. for second-order ODE with three-coeff priors:
        #   output_idx = 2
        #   num_tcoeffs = 3 (state, position, velocity)
        #   -> lift_by = 0, because all states are observed directly
        # For second-order ODE with five-coeff priors:
        #   output_idx = 2
        #   num_tcoeffs = 5
        #   -> lift_by = 2, because without jet extension,
        #   the two highest derivatives remain unobserved
        lift_by = num_tcoeffs - output_idx - 1
        return self.jet_lift(lift_by=lift_by)

    def jet_lift(self, *, lift_by: int) -> "JetOde":
        """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
        if not isinstance(lift_by, int):
            raise TypeError

        if len(self.tcoeff_indices_output) != 1:
            msg = "Jet-lifting is only implemented for ODEs with a single output Taylor coefficient."
            raise NotImplementedError(msg)

        vf_lifted = self.lift(self.vector_field, lift_by=lift_by)
        order = self.num_tcoeffs_in_args + lift_by

        [idx] = self.tcoeff_indices_output
        tcoeff_indices_output = [idx + ell for ell in range(lift_by + 1)]
        return JetOde(
            vf_lifted,
            num_tcoeffs_in_args=order,
            jacobian=self.jacobian,
            tcoeff_indices_output=tcoeff_indices_output,
        )


class JetOdeAutonomous(_JetOdeCommon, Generic[T]):
    """An autonomous ODE u^(k) = f(u, u', ...) where f does not depend on t."""

    def __init__(
        self,
        autonomous,
        jacobian: jacobians.Jacobian,
        num_tcoeffs_in_args: int,
        tcoeff_indices_output: list[int],
    ):
        def vector_field(*, jet_coords, t):
            del t
            return autonomous(jet_coords=jet_coords)

        super().__init__(
            vector_field,
            jacobian=jacobian,
            num_tcoeffs_in_args=num_tcoeffs_in_args,
            tcoeff_indices_output=tcoeff_indices_output,
        )
        self.autonomous = autonomous

    def jet_lift_max(self, *, num_tcoeffs: int):
        raise NotImplementedError

    def jet_lift(self, *, lift_by: int):
        raise NotImplementedError


def ode(func: _ProtocolODEFirstOrder, /, *, jacobian: jacobians.Jacobian | None = None):
    """Construct a description of an ODE u' = f(u, t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        [y] = jet_coords
        return [func(y, t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOde(
        jetfunc, jacobian=jacobian, num_tcoeffs_in_args=1, tcoeff_indices_output=[1]
    )


def ode_order_two(
    func: _ProtocolODEOrderTwo, /, *, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an ODE u'' = f(u, u', t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        (y, dy) = jet_coords
        return [func(y, dy, t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOde(
        jetfunc, jacobian=jacobian, num_tcoeffs_in_args=2, tcoeff_indices_output=[2]
    )


# No typing because arbitrary order is difficult to type (unlike ode and ode_order_two)


def ode_order_arbitrary(
    func, /, *, num_tcoeffs_in_args: int, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an ODE of arbitrary order."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        return [func(*jet_coords[:num_tcoeffs_in_args], t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOde(
        jetfunc,
        jacobian=jacobian,
        num_tcoeffs_in_args=num_tcoeffs_in_args,
        tcoeff_indices_output=[num_tcoeffs_in_args],
    )


class _ProtocolODEAutonomous(Protocol[T]):
    def __call__(self, u: T, /) -> T: ...


def ode_autonomous(
    func: _ProtocolODEAutonomous, /, *, jacobian: jacobians.Jacobian | None = None
):
    """Construct a description of an autonomous ODE u' = f(u)."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        (y,) = jet_coords
        return func(y)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOdeAutonomous(
        autonomous, jacobian=jacobian, num_tcoeffs_in_args=1, tcoeff_indices_output=[1]
    )


class _ProtocolODEAutonomousOrderTwo(Protocol[T]):
    def __call__(self, u: T, du: T, /) -> T: ...


def ode_autonomous_order_two(
    func: _ProtocolODEAutonomousOrderTwo,
    /,
    *,
    jacobian: jacobians.Jacobian | None = None,
):
    """Construct a description of an autonomous ODE u'' = f(u, u')."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        (y, dy) = jet_coords
        return func(y, dy)

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOdeAutonomous(
        autonomous, jacobian=jacobian, num_tcoeffs_in_args=2, tcoeff_indices_output=[2]
    )


# No typing because arbitrary order is difficult to type (unlike ode and ode_order_two)


def ode_autonomous_order_arbitrary(
    func, /, *, num_tcoeffs_in_args: int, jacobian: jacobians.Jacobian | None = None
) -> "JetOdeAutonomous":
    """Construct an autonomous ODE of arbitrary order."""

    def autonomous(*, jet_coords: Sequence[T]) -> T:
        return func(*jet_coords[:num_tcoeffs_in_args])

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetOdeAutonomous(
        autonomous,
        jacobian=jacobian,
        num_tcoeffs_in_args=num_tcoeffs_in_args,
        tcoeff_indices_output=[num_tcoeffs_in_args],
    )


class JetResidual(JetAbstract):
    """A residual on jet coordinates, ie a function that operates on (u, u', ..., t)."""

    def __init__(
        self, residual_function, jacobian: jacobians.Jacobian, num_tcoeffs_in_args: int
    ):
        super().__init__(jacobian=jacobian, num_tcoeffs_in_args=num_tcoeffs_in_args)
        self.residual_function = residual_function

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> Any:
        """Make the vector field callable like the original user function to hide the "sophisticated" API."""
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.residual_function(jet_coords=jet_coords, t=t)

    def jet_lift_max(self, *, num_tcoeffs: int) -> "JetResidual":
        return self.jet_lift(lift_by=num_tcoeffs - self.num_tcoeffs_in_args)

    def jet_lift(self, *, lift_by: int) -> "JetResidual":
        """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
        if not isinstance(lift_by, int):
            raise TypeError
        residual_lifted = self.lift(self.residual_function, lift_by=lift_by)
        order = self.num_tcoeffs_in_args + lift_by
        return JetResidual(
            residual_lifted, num_tcoeffs_in_args=order, jacobian=self.jacobian
        )


class _ProtocolResidualPosition(Protocol[T_contra]):
    def __call__(self, u: T_contra, /, *, t: float) -> Any: ...


def residual_position(
    func: _ProtocolResidualPosition, /, *, jacobian: jacobians.Jacobian | None = None
) -> JetResidual:
    """Construct a description of a residual f(u, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        (y,) = jet_coords
        return [func(y, t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetResidual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=1)


def residual_from_stack(*residual_stack: *tuple[JetResidual, ...]) -> JetResidual:
    """Construct a description of a residual by stacking other residuals."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        return [
            r.residual_function(jet_coords=jet_coords[: r.num_tcoeffs_in_args], t=t)
            for r in residual_stack
        ]

    nums = [r.num_tcoeffs_in_args for r in residual_stack]
    num_args = max(nums)
    jacobian = residual_stack[0].jacobian
    return JetResidual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=num_args)


class _ProtocolResidualVelocity(Protocol[T_contra]):
    def __call__(self, u: T_contra, du: T_contra, /, *, t: float) -> Any: ...


def residual_velocity(
    func: _ProtocolResidualVelocity, /, *, jacobian: jacobians.Jacobian | None = None
) -> JetResidual:
    """Construct a description of a residual f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        (y, dy) = jet_coords
        return [func(y, dy, t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetResidual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=2)


class _ProtocolResidualAcceleration(Protocol[T_contra]):
    def __call__(
        self, u: T_contra, du: T_contra, ddu: T_contra, /, *, t: float
    ) -> Any: ...


def residual_acceleration(
    func: _ProtocolResidualAcceleration,
    /,
    *,
    jacobian: jacobians.Jacobian | None = None,
) -> JetResidual:
    """Construct a description of a residual f(u, du, ddu, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> list[T]:
        (y, dy, ddy) = jet_coords
        return [func(y, dy, ddy, t=t)]

    if jacobian is None:
        jacobian = jacobians.jacobian_monte_carlo_rev()

    return JetResidual(jetfunc, jacobian=jacobian, num_tcoeffs_in_args=3)


def residual_from_ode(ode: JetOde) -> JetResidual:
    """Construct a JetResidual from a JetOde.

    The residual is u^(k) - f(u, u', ..., t) = 0.
    """

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        output = [jet_coords[i] for i in ode.tcoeff_indices_output]
        inputs = jet_coords[: ode.num_tcoeffs_in_args]
        vf_eval = ode.vector_field(jet_coords=inputs, t=t)
        return tree.tree_map(lambda a, b: a - b, output, vf_eval)

    return JetResidual(
        jetfunc, jacobian=ode.jacobian, num_tcoeffs_in_args=ode.num_tcoeffs_in_args + 1
    )
