"""Utilities for probabilistic ODE solver implementations."""

from probdiffeq.backend import func, linalg, np, structs, tree
from probdiffeq.backend.typing import Array, Generic, TypeVar
from probdiffeq.util import cholesky_util

S = TypeVar("S")
"""A type-variable to describe interpolation results."""


__all__ = [
    "InterpResult",
    "preconditioner_taylor",
    "system_matrices_1d_iwp",
    "verify_taylor_coefficient_pytree",
]


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


def system_matrices_1d_iwp(num_derivatives):
    """Construct the system matrices of the integrated Wiener process."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...

    # Cholesky factor of flip(hilbert(n))
    Q_1d = cholesky_util.cholesky_hilbert(num_derivatives + 1)
    Q_1d_flipped = np.flip(Q_1d, axis=0)
    Q_1d = linalg.qr_r(Q_1d_flipped.T).T

    scale = np.sign(linalg.diagonal(Q_1d))
    scale = np.where(scale == 0.0, 1.0, scale)
    Q_1d = Q_1d * scale[None, :]
    return A_1d, Q_1d


def preconditioner_taylor(num_derivatives):
    """Construct the diagonal preconditioner for Taylor-coefficient state-spaces."""
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)

    def precon(dt):
        scaling_vector = np.power(dt, powers) / scales
        scaling_vector_inv = np.power(dt, -powers) * scales
        return scaling_vector, scaling_vector_inv

    return precon


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = func.vmap(k, in_axes=(0, None), out_axes=-1)
    return func.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))


def verify_taylor_coefficient_pytree(x, /):
    if isinstance(x, Array):
        msg = "Mean must be a pytree, not an array."
        raise TypeError(msg)

    try:
        x_list = [*x]
        shape0 = tree.tree_map(np.shape, x_list[0])
    except Exception as error:
        msg = "Mean must be a pytree of the form [M_1, ..., M_{num_coeffs}], "
        msg += "where each M_i is a pytree of the same structure."
        msg += f" Received: {x}."
        raise ValueError(msg) from error

    for i, x_i in enumerate(x_list):
        xi_shape = tree.tree_map(np.shape, x_i)
        if xi_shape != shape0:
            msg = "All leaves of the mean must have the same shape."
            msg += f" However, leaf {i} has shape {xi_shape}"
            msg += f", while leaf 0 has shape {shape0}"
            raise ValueError(msg)
