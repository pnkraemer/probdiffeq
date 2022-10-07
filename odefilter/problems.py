"""Problem types."""

from typing import Callable, Tuple

import equinox as eqx
from jaxtyping import Array, Float, PyTree

# todo: make private and never really show to the end-user?


class InitialValueProblem(eqx.Module):
    """Initial value problem."""

    vector_field: Callable[..., Float[Array, " d"]]
    """ODE function. Signature ``f(*initial_values, t0, *parameters)``."""

    # todo: Make into a tuple of initial values?
    initial_values: Tuple[Float[Array, " d"], ...]
    r"""Initial values.

    A tuple of arrays: one of
    $(u_0,)$, $(u_0, \dot u_0)$, $(u_0, \dot u_0, \ddot u_0)$, et cetera.
    """

    t0: float
    """Initial time-point."""

    t1: float
    """Terminal time-point. Optional."""

    parameters: PyTree
    """Pytree of parameters."""
