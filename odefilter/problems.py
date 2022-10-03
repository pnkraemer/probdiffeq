"""Problem types."""

from typing import Any, Callable, Iterable, NamedTuple, Optional, Union


class InitialValueProblem(NamedTuple):
    f: Callable
    """ODE vector field."""

    y0: Union[Any, Iterable[Any]]
    """Initial values."""

    p: Any
    """Parameters of the initial value problem."""

    t0: float
    """Initial time-point."""

    t1: float
    """Terminal time-point."""

    jac: Optional[Callable] = None
    """Jacobian of the vector field."""
