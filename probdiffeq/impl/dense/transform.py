"""Random variable transformations."""
from typing import Callable

import jax

from probdiffeq import _sqrt_util
from probdiffeq.backend import containers
from probdiffeq.impl import _cond_util, _transform
from probdiffeq.impl.dense import _normal


class Transformation(containers.NamedTuple):
    matmul: Callable
    bias: jax.Array


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        cholesky_new = _sqrt_util.triu_via_qr((A @ rv.cholesky).T).T
        return _normal.Normal(A @ rv.mean + b, cholesky_new)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky).T, R_X=cholesky.T
        )

        # Gather terms and return
        m_cor = mean - gain @ (A @ mean + b)
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(A @ mean + b, r_obs.T)
        return observed, _cond_util.Conditional(gain, corrected)
