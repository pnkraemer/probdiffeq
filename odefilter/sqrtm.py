"""Square-root matrix utility functions."""

import jax.numpy as jnp


def sum_of_sqrtm_factors(*, R1, R2):
    """Compute Cholesky factor of R1.T @ R1 + R2.T @ R2."""
    R = jnp.vstack((R1, R2))
    cov_sqrtm = sqrtm_to_cholesky(R=R)
    if R1.ndim == 0:
        return cov_sqrtm.reshape(())
    return cov_sqrtm


def sqrtm_to_cholesky(*, R):
    """Transform a matrix square root to a Cholesky factor."""
    upper_sqrtm = jnp.linalg.qr(R, mode="r")
    return _make_diagonal_positive(R=upper_sqrtm)


def _make_diagonal_positive(*, R):
    s = jnp.sign(jnp.diag(R))
    x = jnp.where(s == 0, 1, s)
    return x[..., None] * R
