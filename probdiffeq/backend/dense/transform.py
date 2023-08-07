from typing import Callable

import jax

from probdiffeq import _sqrt_util
from probdiffeq.backend import _transform, containers
from probdiffeq.backend.dense import random


class Transformation(containers.NamedTuple):
    matmul: Callable
    bias: jax.Array


class TransformBackEnd(_transform.TransformBackEnd):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        cholesky_new = _sqrt_util.triu_via_qr(A(rv.cholesky).T).T
        return random.Normal(A(rv.mean) + b, cholesky_new)

    def revert(self, rv, transformation, /):
        matmul, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=matmul(cholesky).T, R_X=cholesky.T
        )

        # Gather terms and return
        m_cor = mean - gain @ (matmul(mean) + b)
        corrected = random.Normal(m_cor, r_cor.T)
        observed = random.Normal(matmul(mean) + b, r_obs.T)
        return observed, (corrected, gain)
