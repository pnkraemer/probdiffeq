"""Integrated Brownian motion (IBM) utilities."""

from probdiffeq.backend import functools, linalg, tree_util
from probdiffeq.backend import numpy as np


def system_matrices_1d(num_derivatives, output_scale):
    """Construct the IBM system matrices."""
    x = np.arange(0, num_derivatives + 1)

    A_1d = np.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = np.flip(_hilbert(x))
    return A_1d, output_scale * linalg.cholesky_factor(Q_1d)


def preconditioner_diagonal(dt, *, scales, powers):
    """Construct the diagonal IBM preconditioner."""
    dt_abs = np.abs(dt)
    scaling_vector = np.power(dt_abs, powers) / scales
    scaling_vector_inv = np.power(dt_abs, -powers) * scales
    return scaling_vector, scaling_vector_inv


def preconditioner_prepare(*, num_derivatives):
    powers = np.arange(num_derivatives, -1.0, step=-1.0)
    scales = np.factorial(powers)
    powers = powers + 0.5
    return tree_util.Partial(preconditioner_diagonal, scales=scales, powers=powers)


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = functools.vmap(k, in_axes=(0, None), out_axes=-1)
    return functools.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return np.factorial(n) / (np.factorial(n - k) * np.factorial(k))
