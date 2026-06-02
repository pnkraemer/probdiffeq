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
>>> print(ode(f))
JetFunction(num_derivatives_in_args=1)

Among other things, the vector field wrappers ensure that all internal representations
of the ODEs have the same signature, which means that sometimes (eg for first-order problems),
the internal representation does not match the user-specified one.

>>> print(inspect.signature(f))
(y, *, t)
>>> print(inspect.signature(ode(f)))
(*, u: collections.abc.Sequence[~T], t) -> ~T

This API difference is more pronounced for higher-order problems:

>>> def f_second_order(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(ode(f_second_order))
JetFunction(num_derivatives_in_args=2)
>>>
>>> print(inspect.signature(f_second_order))
(y, dy, /, *, t)
>>> print(inspect.signature(ode(f)))
(*, u: collections.abc.Sequence[~T], t) -> ~T

"""

from probdiffeq._probdiffeq import jacobians, utilities
from probdiffeq.backend import func, inspect, tree
from probdiffeq.backend.typing import Array, Callable, Sequence, TypeVar

T = TypeVar("T")

__all__ = ["DAESystem", "JetFunction", "dae", "implicit", "jet_lift", "ode"]


class JetFunction:
    """A jet function, ie a function that operates on jet coordinates (y, y', ..., t).

    This is typically used to define right-hand sides of (high-order) ODEs
    and residuals in implicit differential equations.
    """

    def __init__(
        self, func, /, jacobian: jacobians.JacobianHandler, num_derivatives_in_args: int
    ):
        self._func = func

        self.jacobian = jacobian
        self.num_derivatives_in_args = num_derivatives_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives_in_args={self.num_derivatives_in_args}, jacobian={self.jacobian})"

    def __call__(self, *, jet_coords: Sequence[T], t: Array) -> T:
        # jet_coords = (u, u', u'', ..., u^(K))
        return self._func(jet_coords=jet_coords, t=t)

    @classmethod
    def from_callable(
        cls, func: Callable[[*Sequence[T], Array], T], /, *, jacobian=None
    ):
        num_derivatives_in_args = _verify_vector_field_signature_and_parse_order(func)

        def jet_function(*, jet_coords: Sequence[T], t: Array) -> T:
            return func(*jet_coords, t=t)

        if jacobian is None:
            jacobian = jacobians.jacobian_hutchinson_fwd()

        return cls(
            jet_function,
            num_derivatives_in_args=num_derivatives_in_args,
            jacobian=jacobian,
        )


def ode(func: Callable[[*Sequence[T], float], T], /, *, jacobian=None) -> JetFunction:
    """A description of an ordinary differential equation y^{(K)} = f(y, y', ..., y^{(K-1)}, t).

    Parameters
    ----------
    func
        A function with the signature f(u: T, *, t) -> T, f(u: T, du: T, *, t) -> T, and so on.
        This is a vector field on jet coordinates with fixed order, and internally,
        probdiffeq represents them as general jet functions
        (because implicit differential equations share most of the source code).
        The number of positional arguments determines the order of the jet
        -- think, the number of derivatives in the argument list --, and
        is read from the vector field signature automatically.
        For example, if we pass f(u, t), the order is one; if we pass f(u, du, t), the order is two.
    jacobian
        How to handle Jacobians of the vector field. For example, whether forward or reverse mode
        should be used, or how we should extract traces and diagonals from Jacobian-vector products.
    """
    return JetFunction.from_callable(func, jacobian=jacobian)


def implicit(
    func: Callable[[*Sequence[T], float], T], /, *, jacobian=None
) -> JetFunction:
    """A description of an implicit differential equation f(u, du, t) = 0."""
    # No implementation difference between ode and implicit, but
    # we don't want to force the user to think in terms of jet functions
    # so we offer these wrappers.
    return JetFunction.from_callable(func, jacobian=jacobian)


class DAESystem:
    """Differential-algebraic equations."""

    def __init__(self, differential: JetFunction, algebraic: JetFunction):
        self.differential = differential
        self.algebraic = algebraic

    def __repr__(self):
        return f"{self.__class__.__name__}(differential={self.differential}, algebraic={self.algebraic})"


def dae(differential, algebraic):
    return DAESystem(differential, algebraic)


def jet_lift(jetfun: JetFunction, lift_by: int):
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
            fx = jetfun(jet_coords=y_tree, t=t)
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
    return JetFunction(
        jetfun_lifted, num_derivatives_in_args=order, jacobian=jetfun.jacobian
    )


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
