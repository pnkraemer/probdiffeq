"""Tests for square-root matrices.

These are so crucial and hard to debug that they need their own test set.
"""
import jax.numpy as jnp

from odefilter.implementations import _sqrtm


def test_revert_kernel():

    C = jnp.arange(9.0).reshape((3, 3))
    H = jnp.arange(1.0, 13.0).reshape((4, 3))
    X = jnp.eye(4) + jnp.arange(16.0).reshape((4, 4))

    S = H @ C @ C.T @ H.T + X @ X.T
    K = C @ C.T @ H.T @ jnp.linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = _sqrtm.revert_gauss_markov_correlation(
        R_X_F=C.T @ H.T, R_X=C.T, R_YX=X.T
    )

    def cov(x):
        return x.T @ x

    assert jnp.allclose(cov(extra), S)
    assert jnp.allclose(g, K)
    assert jnp.allclose(cov(bw_noise), C1)


def test_revert_kernel_noisefree():

    C = jnp.arange(9.0).reshape((3, 3))
    H = jnp.arange(1.0, 7.0).reshape((2, 3))

    S = H @ C @ C.T @ H.T
    K = C @ C.T @ H.T @ jnp.linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = _sqrtm.revert_gauss_markov_correlation_noisefree(
        R_X_F=C.T @ H.T, R_X=C.T
    )

    def cov(x):
        return x.T @ x

    assert jnp.allclose(cov(extra), S)
    assert jnp.allclose(g, K)
    assert jnp.allclose(cov(bw_noise), C1)
