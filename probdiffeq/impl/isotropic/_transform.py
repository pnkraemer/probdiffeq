from probdiffeq.backend import numpy as np
from probdiffeq.impl import _transform
from probdiffeq.impl.isotropic import _normal
from probdiffeq.util import cholesky_util, cond_util


class TransformBackend(_transform.TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky
        cholesky_new = cholesky_util.triu_via_qr((A @ cholesky).T)
        cholesky_squeezed = np.reshape(cholesky_new, ())
        return _normal.Normal((A @ mean) + b, cholesky_squeezed)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree
        #   to revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky).T, R_X=cholesky.T
        )
        cholesky_obs = np.reshape(r_obs, ())
        cholesky_cor = r_cor.T

        # Gather terms and return
        mean_observed = A @ mean + b
        m_cor = mean - gain * mean_observed
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)
