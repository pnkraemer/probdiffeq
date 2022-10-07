"""Problem types."""

from dataclasses import dataclass
from typing import Callable, Tuple

import jax.tree_util
from jaxtyping import Array, Float, PyTree

# todo: make private and never really show to the end-user?


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InitialValueProblem:
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

    def tree_flatten(self):
        """Flatten the data structure."""
        aux_data = self.vector_field
        children = (self.initial_values, self.t0, self.t1, self.parameters)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the data structure."""
        vector_field = aux_data
        initial_values, t0, t1, parameters = children
        return cls(
            vector_field=vector_field,
            initial_values=initial_values,
            t0=t0,
            t1=t1,
            parameters=parameters,
        )
