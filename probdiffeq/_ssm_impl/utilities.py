from probdiffeq.backend import func, linalg, np, tree
from probdiffeq.backend.typing import Array
from probdiffeq.util import cholesky_util

__all__ = [
    "preconditioner_taylor",
    "system_matrices_1d_iwp",
    "verify_taylor_coefficient_pytree",
]


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


def preconditioner_taylor(*, num_derivatives):
    """Construct the diagonal preconditioner for Taylor-coefficient state-spaces."""
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)
    powers = powers + 0.5

    def precon(dt):
        dt_abs = np.abs(dt)
        scaling_vector = np.power(dt_abs, powers) / scales
        scaling_vector_inv = np.power(dt_abs, -powers) * scales
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
