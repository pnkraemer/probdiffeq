from typing import Callable

import jax

from probdiffeq import _sqrt_util
from probdiffeq.backend import _cond, containers
from probdiffeq.backend.dense import random


class Conditional(containers.NamedTuple):
    matmul: Callable
    noise: random.Normal


class Transformation(containers.NamedTuple):
    matmul: Callable
    bias: jax.Array


class TransformImpl(_cond.TransformImpl):
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


class ConditionalImpl(_cond.ConditionalImpl):
    def apply(self, x, conditional, /):
        matrix, noise = conditional
        return random.Normal(matrix @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        matmul, noise = conditional
        R_stack = ((matmul @ rv.cholesky).T, noise.cholesky.T)
        cholesky_new = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
        return random.Normal(matmul @ rv.mean + noise.mean, cholesky_new)

    def merge(self, cond1, cond2, /):
        raise NotImplementedError

    def revert(self, rv, conditional, /):
        matrix, noise = conditional
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
            R_X_F=(matrix @ cholesky).T, R_X=cholesky.T, R_YX=noise.cholesky.T
        )

        # Gather terms and return
        mean_observed = matrix @ mean + noise.mean
        m_cor = mean - gain @ mean_observed
        corrected = random.Normal(m_cor, r_cor.T)
        observed = random.Normal(mean_observed, r_obs.T)
        return observed, (corrected, gain)


class ConditionalBackEnd(_cond.ConditionalBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    @property
    def transform(self) -> TransformImpl:
        return TransformImpl()

    @property
    def conditional(self) -> ConditionalImpl:
        return ConditionalImpl()
