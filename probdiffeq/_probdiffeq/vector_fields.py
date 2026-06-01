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
VectorField(order=1)

Among other things, the vector field wrappers ensure that all internal representations
of the ODEs have the same signature, which means that sometimes (eg for first-order problems),
the internal representation does not match the user-specified one.

>>> print(inspect.signature(f))
(y, *, t)
>>> print(inspect.signature(ode(f)))
(y_and_dy_and_ddy_etc: collections.abc.Sequence[~T], /, *, t) -> ~T

This API difference is more pronounced for higher-order problems:

>>> def f_2nd(y, dy, /, *, t):
...     return y + dy
>>>
>>> print(ode(f_2nd))
VectorField(order=2)
>>>
>>> print(inspect.signature(f_2nd))
(y, dy, /, *, t)
>>> print(inspect.signature(ode(f)))
(y_and_dy_and_ddy_etc: collections.abc.Sequence[~T], /, *, t) -> ~T

"""

from probdiffeq.backend import inspect
from probdiffeq.backend.typing import Callable, Sequence, TypeVar

T = TypeVar("T")

__all__ = ["DAESystem", "VectorField", "dae", "ode"]


class VectorField:
    """A vector field, i.e. a function that computes the right-hand side of an ODE, with the correct API for use in probdiffeq."""

    def __init__(self, func, num_derivatives_in_args: int):
        self._func = func
        self.num_derivatives_in_args = num_derivatives_in_args

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives_in_args={self.num_derivatives_in_args})"

    def __call__(self, *, u: Sequence[T], t) -> T:
        return self._func(u=u, t=t)


class DAESystem:
    def __init__(self, differential: VectorField, algebraic: VectorField):
        assert (
            differential.num_derivatives_in_args
            == algebraic.num_derivatives_in_args + 1
        )
        self.differential = differential
        self.algebraic = algebraic

    def __repr__(self):
        return f"{self.__class__.__name__}(differential={self.differential}, algebraic={self.algebraic})"


def ode(func: Callable[[*Sequence[T], float], T], /) -> VectorField:
    """Convert a function that computes the right-hand side of an ODE into a vector field with the correct API."""
    num_derivatives_in_args = _verify_vector_field_signature_and_parse_order(func)

    def wrapper(u: Sequence[T], t) -> T:
        return func(*u, t=t)

    return VectorField(wrapper, num_derivatives_in_args=num_derivatives_in_args)


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


def dae(
    differential: Callable[[*Sequence[T], float], T],
    algebraic: Callable[[*Sequence[T], float], T],
) -> VectorField:
    return DAESystem(differential=ode(differential), algebraic=ode(algebraic))
