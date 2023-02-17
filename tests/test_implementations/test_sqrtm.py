"""Tests for square-root matrices.

These are so crucial and annoying to debug that they need their own test set.
"""
from math import prod

import jax.numpy as jnp
import pytest_cases

from probdiffeq.implementations import _sqrtm

_SHAPES = ([(4, 3), (3, 3), (4, 4)], [(2, 3), (3, 3), (2, 2)])


@pytest_cases.parametrize("HCshape, Cshape, Xshape", _SHAPES)
def test_revert_kernel_scalar(HCshape, Cshape, Xshape):
    HC = _some_array(HCshape) + 1.0
    C = _some_array(Cshape) + 2.0
    X = _some_array(Xshape) + 3.0 + jnp.eye(*Xshape)

    S = HC @ HC.T + X @ X.T
    K = C @ HC.T @ jnp.linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = _sqrtm.revert_conditional(R_X_F=HC.T, R_X=C.T, R_YX=X.T)

    def cov(x):
        return x.T @ x

    assert jnp.allclose(cov(extra), S)
    assert jnp.allclose(g, K)
    assert jnp.allclose(cov(bw_noise), C1)


@pytest_cases.parametrize("Cshape, HCshape", ([(3, 3), (2, 3)],))
def test_revert_kernel_noisefree(Cshape, HCshape):
    C = _some_array(Cshape) + 1.0
    HC = _some_array(HCshape) + 2.0

    S = HC @ HC.T
    K = C @ HC.T @ jnp.linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = _sqrtm.revert_conditional_noisefree(R_X_F=HC.T, R_X=C.T)

    def cov(x):
        return x.T @ x

    assert jnp.allclose(cov(extra), S)
    assert jnp.allclose(g, K)
    assert jnp.allclose(cov(bw_noise), C1)


def _some_array(shape):
    return jnp.arange(1.0, 1.0 + prod(shape)).reshape(shape)
