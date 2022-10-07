"""Problem types."""

from typing import Any, Callable, Iterable, NamedTuple, Optional, Union

import jax.tree_util
from jaxtyping import Array, Float


@jax.tree_util.register_pytree_node_class
class InitialValueProblem(NamedTuple):
    """Initial value problem."""

    vector_field: Callable[..., Float[Array, " d"]]
    """ODE function. Signature f(*initial_values, t0, *parameters)."""

    # todo: Make into a tuple of initial values?
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

    parameters: Any
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
