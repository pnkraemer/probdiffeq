"""Random variable transformations."""
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _transform
from probdiffeq.impl.scalar import _normal
from probdiffeq.util import cholesky_util, cond_util


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        # currently, assumes that A(rv.cholesky) is a vector, not a matrix.
        matmul, b = transformation
        cholesky_new = cholesky_util.triu_via_qr(matmul(rv.cholesky)[:, None])
        cholesky_new_squeezed = np.reshape(cholesky_new, ())
        return _normal.Normal(matmul(rv.mean) + b, cholesky_new_squeezed)

    def revert(self, rv, transformation, /):
        # Assumes that A maps a vector to a scalar...

        # Extract information
        A, b = transformation

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree
        #  to transformation_revert_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=A(rv.cholesky)[:, None], R_X=rv.cholesky.T
        )
        cholesky_obs = np.reshape(r_obs, ())
        cholesky_cor = r_cor.T
        gain = np.squeeze_along_axis(gain, axis=-1)

        # Gather terms and return
        m_cor = rv.mean - gain * (A(rv.mean) + b)
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(A(rv.mean) + b, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)
