"""Random variable transformations."""
from probdiffeq.backend import containers
from probdiffeq.backend.typing import Array, Callable
from probdiffeq.impl import _transform
from probdiffeq.impl.dense import _normal
from probdiffeq.util import cholesky_util, cond_util


class Transformation(containers.NamedTuple):
    matmul: Callable
    bias: Array


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        cholesky_new = cholesky_util.triu_via_qr((A @ rv.cholesky).T).T
        return _normal.Normal(A @ rv.mean + b, cholesky_new)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky).T, R_X=cholesky.T
        )

        # Gather terms and return
        m_cor = mean - gain @ (A @ mean + b)
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(A @ mean + b, r_obs.T)
        return observed, cond_util.Conditional(gain, corrected)
