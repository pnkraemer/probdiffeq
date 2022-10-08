"""Square-root matrix utility functions."""

import jax.numpy as jnp


def revert_markov_kernel(*, h_matmul_c_sqrtm, c_sqrtm, r_sqrtm):

    output_dim, input_dim = h_matmul_c_sqrtm.shape

    blockmat = jnp.block(
        [
            [r_sqrtm, h_matmul_c_sqrtm],
            [jnp.zeros((input_dim, output_dim)), c_sqrtm],
        ]
    ).T

    R = jnp.linalg.qr(blockmat, mode="r")

    R1 = R[:output_dim, :output_dim]  # observed RV
    R12 = R[:output_dim, output_dim:]  # something like the crosscov
    R3 = R[output_dim:, output_dim:]  # corrected RV

    # todo: what is going on here???
    #  why lstsq? The matrix should be well-conditioned.
    gain = jnp.linalg.lstsq(R1, R12)[0].T

    c_sqrtm_cor = _make_diagonal_positive(R=R3).T
    c_sqrtm_obs = _make_diagonal_positive(R=R1).T
    return c_sqrtm_obs, (c_sqrtm_cor, gain)


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
