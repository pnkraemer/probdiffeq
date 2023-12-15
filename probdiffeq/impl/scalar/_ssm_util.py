"""SSM utilities."""
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _ssm_util
from probdiffeq.impl.scalar import _normal
from probdiffeq.util import cholesky_util, cond_util, ibm_util


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = np.stack(tcoeffs)
        m0_corrected = np.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = np.zeros((num_derivatives + 1, num_derivatives + 1))
        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return _normal.Normal(p * rv.mean, p[:, None] * rv.cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        A = p[:, None] * A * p_inv[None, :]
        noise = _normal.Normal(p * noise.mean, p[:, None] * noise.cholesky)
        return cond_util.Conditional(A, noise)

    def ibm_transitions(self, num_derivatives, output_scale=1.0):
        a, q_sqrtm = ibm_util.system_matrices_1d(num_derivatives, output_scale)
        q0 = np.zeros((num_derivatives + 1,))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return cond_util.Conditional(a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        transition = np.eye(ndim)
        mean = np.zeros((ndim,))
        cov_sqrtm = np.zeros((ndim, ndim))
        noise = _normal.Normal(mean, cov_sqrtm)
        return cond_util.Conditional(transition, noise)

    def standard_normal(self, ndim, /, output_scale):
        mean = np.zeros((ndim,))
        cholesky = output_scale * np.eye(ndim)
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)
