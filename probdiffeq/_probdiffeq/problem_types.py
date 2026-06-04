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
>>> @ode
... def f(y, *, t):
...     return y
>>>
>>> print(f)
ODEFunction(num_derivatives_in_args=1, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))


Higher-order problems:

>>> @ode_second_order
... def f(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(f)
ODEFunction(num_derivatives_in_args=2, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))

General constraints:

>>> import jax.numpy as jnp
>>>
>>> @residual_state
... def g(y, /, *, t):
...     return jnp.abs2(y)
>>>
>>> print(g)
Residual(num_derivatives_in_args=1, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))

Higher-order constraints:

>>> @residual_state_velocity
... def g(y, /, dy, *, t):
...     return jnp.abs2(dy)
>>>
>>> print(g)
Residual(num_derivatives_in_args=2, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10))



"""

from probdiffeq._probdiffeq import utilities
from probdiffeq.backend import func, linalg, random, tree
from probdiffeq.backend.typing import Any, Array, Generic, Protocol, Sequence, TypeVar

__all__ = [
    "JacobianHandler",
    "ODEFunction",
    "ProtocolODEFirstOrder",
    "ProtocolODESecondOrder",
    "ProtocolResidualState",
    "ProtocolResidualVelocity",
    "Residual",
    "dae_system",
    "jacobian_hutchinson_fwd",
    "jacobian_hutchinson_rev",
    "jacobian_materialize",
    "jet_lift",
    "ode",
    "ode_second_order",
    "residual_state",
    "residual_state_velocity",
    "residual_state_velocity_acceleration",
]


class JacobianHandler:
    """An interface for working with Jacobian matrices."""

    def init_jacobian_handler(self):
        """Initialize the handler state.

        For example, if the handler uses stochastic sampling,
        this initialisation would create a random key.
        """
        raise NotImplementedError

    def materialize_dense(self, fun, x, state, /):
        """Materialize a dense Jacobian.

        This is typically used for first-order linearization in dense
        state-space models.
        """
        raise NotImplementedError

    def calculate_trace(self, fun, x, state, /):
        """Calculate the trace of a Jacobian.

        This is typically used for first-order linearization in isotropic
        state-space models.
        """
        raise NotImplementedError

    def calculate_diagonal(self, fun, x, state, /):
        """Calculate the diagonal of a Jacobian.

        This is typically used for first-order linearization in block-diagonal
        state-space models.
        """
        raise NotImplementedError


class jacobian_materialize(JacobianHandler):
    """Construct a handler that always materialized Jacobian matrices.

    Use this Jacobian if the dimension of the problem is relatively small.
    """

    def __init__(self, *, jacfun=func.jacfwd) -> None:
        self.jacfun = jacfun

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(jacfun={self.jacfun})"

    def init_jacobian_handler(self):
        return ()

    def materialize_dense(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, ()

    def calculate_trace(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_trace = linalg.trace(dfx)
        return fx, dfx_trace, ()

    def calculate_diagonal(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_diagonal = linalg.diagonal(dfx)
        return fx, dfx_diagonal, ()


class jacobian_hutchinson_fwd(JacobianHandler):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **forward-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seed={self.seed}, num_probes={self.num_probes})"
        )

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, Jvp(s)))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        vJv = func.vmap(lambda s: s * Jvp(s))(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


class jacobian_hutchinson_rev(JacobianHandler):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **reverse-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seed={self.seed}, num_probes={self.num_probes})"
        )

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacrev(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, vjp(s)[0]))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        vJv = func.vmap(lambda s: s * vjp(s)[0])(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


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

    def __init__(self, jacobian: JacobianHandler, num_derivatives_in_args: int):
        self.jacobian = jacobian
        self.num_derivatives_in_args = num_derivatives_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives_in_args={self.num_derivatives_in_args}, jacobian={self.jacobian})"


class ProtocolODEFirstOrder(Protocol[T]):
    def __call__(self, u: T, /, *, t: float) -> T: ...


class ProtocolODESecondOrder(Protocol[T]):
    def __call__(self, u: T, du: T, /, *, t: float) -> T: ...


class ODEFunction(_AbstractJetFunction, Generic[T]):
    def __init__(
        self, vector_field, jacobian: JacobianHandler, num_derivatives_in_args: int
    ):
        super().__init__(
            jacobian=jacobian, num_derivatives_in_args=num_derivatives_in_args
        )
        self.vector_field = vector_field

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> T:
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.vector_field(jet_coords=jet_coords, t=t)


def ode(func: ProtocolODEFirstOrder, /, *, jacobian: JacobianHandler | None = None):
    """Construct a description of an  ODE y' = f(y, t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    return ODEFunction(jetfunc, jacobian=jacobian, num_derivatives_in_args=1)


def ode_second_order(
    func: ProtocolODESecondOrder, /, *, jacobian: JacobianHandler | None = None
):
    """Construct a description of an  ODE y'' = f(y, y', t)."""

    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    return ODEFunction(jetfunc, jacobian=jacobian, num_derivatives_in_args=2)


class Residual(_AbstractJetFunction):
    """A residual on jet coordinates, ie a function that operates on (y, y', ..., t)."""

    def __init__(
        self, residual_function, jacobian: JacobianHandler, num_derivatives_in_args: int
    ):
        super().__init__(
            jacobian=jacobian, num_derivatives_in_args=num_derivatives_in_args
        )
        self.residual_function = residual_function

    def __call__(self, *jet_coords: *tuple[T], t: Array) -> Any:
        """Make the vector field callable like the original user function to hide the "sophisticated" API."""
        # jet_coords = (u(t), u'(t), u''(t), ..., u^(K)(t))
        return self.residual_function(jet_coords=jet_coords, t=t)


class ProtocolResidualState(Protocol[T_contra]):
    def __call__(self, u: T_contra, /, *, t: float) -> Any: ...


def residual_state(
    func: ProtocolResidualState, /, *, jacobian: JacobianHandler | None = None
) -> Residual:
    """Construct a description of a residual f(u, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y,) = jet_coords
        return func(y, t=t)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_derivatives_in_args=1)


class ProtocolResidualVelocity(Protocol[T_contra]):
    def __call__(self, u: T_contra, du: T_contra, /, *, t: float) -> Any: ...


def residual_state_velocity(
    func: ProtocolResidualVelocity, /, *, jacobian: JacobianHandler | None = None
) -> Residual:
    """Construct a description of a residual f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy) = jet_coords
        return func(y, dy, t=t)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_derivatives_in_args=2)


class ProtocolResidualAcceleration(Protocol[T_contra]):
    def __call__(
        self, u: T_contra, du: T_contra, ddu: T_contra, /, *, t: float
    ) -> Any: ...


def residual_state_velocity_acceleration(
    func: ProtocolResidualAcceleration, /, *, jacobian: JacobianHandler | None = None
) -> Residual:
    """Construct a description of a residual f(u, du, t) = 0."""

    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    def jetfunc(*, jet_coords: Sequence[T], t: float) -> T:
        (y, dy, ddy) = jet_coords
        return func(y, dy, ddy, t=t)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    return Residual(jetfunc, jacobian=jacobian, num_derivatives_in_args=3)


class dae_system:
    """Differential-algebraic equations."""

    def __init__(self, differential: Residual, algebraic: Residual):
        if not isinstance(differential, Residual):
            raise TypeError(differential)
        if not isinstance(algebraic, Residual):
            raise TypeError(algebraic)

        self.differential = differential
        self.algebraic = algebraic

    def __repr__(self):
        return f"{self.__class__.__name__}(differential={self.differential}, algebraic={self.algebraic})"


def jet_lift(residual: Residual, lift_by: int) -> Residual:
    """Lift a function on k-jet coordinates to one on (k+m)-jet coordinates."""
    if not isinstance(lift_by, int):
        raise TypeError

    def residual_lifted(*, jet_coords: Sequence[T], t) -> Sequence[T]:
        tcoeffs_all = jet_coords
        _, unravel_one = tree.ravel_pytree(tcoeffs_all[0])

        lift_by_upper = len(tcoeffs_all) - residual.num_derivatives_in_args
        if lift_by < 0 or lift_by > lift_by_upper:
            msg = "The provided jet-order is incompatible with the residual order."
            msg += f" Expected: 0 <= lift_by <= {lift_by_upper}."
            msg += f" Received: lift_by == {lift_by}."
            raise ValueError(msg)
        order = residual.num_derivatives_in_args + lift_by
        tcoeffs = tcoeffs_all[:order]

        # Flatten the residual because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel_one(s) for s in y]
            fx = residual.residual_function(jet_coords=y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]

        ps, ss = utilities.jet_coords_to_primals_and_series(
            flat, residual.num_derivatives_in_args
        )

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)

            # Return a sequence to be compatible with Taylor-coeff logic,
            # but don't bother unflattening the content
            # because the result will be compared to zero anyway
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = residual.num_derivatives_in_args + lift_by
    return Residual(
        residual_lifted, num_derivatives_in_args=order, jacobian=residual.jacobian
    )
