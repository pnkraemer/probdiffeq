"""Utilities for probabilistic ODE solver implementations."""

from probdiffeq.backend import structs, tree
from probdiffeq.backend.typing import Generic, TypeVar

S = TypeVar("S")
"""A type-variable to describe interpolation results."""


__all__ = ["InterpResult"]


@tree.register_dataclass
@structs.dataclass
class InterpResult(Generic[S]):
    """A datastructure to store interpolation results.

    To ensure correct adaptive time-stepping, it is important
    to distinguish step-from variables from interpolate-from variables.

    For some solvers, e.g. fixed-point-smoother-based ones,
    both stepping and interpolating variables are adjusted during interpolation.
    """

    step_from: S
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interp_from: S
    """The new `interp_from` field.

    At time `t`. Use this as the left-most reference state
    in future interpolations.

    The difference between `interpolated` and `interp_from`
    is important around checkpoints:

    - `interpolated` belongs to the just-concluded time interval,
    - `interp_from` belongs to the to-be-started time interval.

    Concretely, this means that for fixed-point smoothers,
    `interp_from` has a unit backward model whereas `interpolated`
    remembers how to step back to the previous target location.
    """


def jet_coords_to_primals_and_series(taylor_series, num, /):
    """Compute Jet-compatible arguments from a Taylor series.

    That is, for a function like f(u, u'), the present function
    turns a Taylor series (u, u', u'', ...) into arguments
    compatible with jax.experimental.jet.

    Arguments
    ---------
    taylor_series
        A sequence of arrays to evaluate the Taylor series at.
    num
        The number of inputs to the root
        (2 for a first-order ODE, 3 for second-order, etc.)

    Examples
    --------
    >>> a = (1, 2, 3, 4, 5)
    >>> print(jet_unpack_series(a, n=1))
    [1], [(1, 2, 3, 4, 5)]
    >>> print(jet_unpack_series(a, n=2))
    [1, 2], [(1, 2, 3, 4), (2, 3, 4, 5)]
    >>> print(jet_unpack_series(a, n=3))
    [1, 2, 3], [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    """
    primals, series = taylor_series[:num], taylor_series[1:]

    def mask(i):
        return None if i == 0 else i

    series_ = [series[mask(k) : mask(k + 1 - num)] for k in range(num)]
    return primals, series_
