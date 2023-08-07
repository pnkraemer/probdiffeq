import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.backend import _ssm_util
from probdiffeq.backend.isotropic import _normal
from probdiffeq.statespace import _ibm_util


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives, output_scale):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives, output_scale)
        q0 = jnp.zeros((num_derivatives + 1,) + self.ode_shape)
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim):
        raise NotImplementedError

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        c_sqrtm0_corrected = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
        m0_corrected = jnp.stack(tcoeffs)
        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return _normal.Normal(p[:, None] * rv.mean, p[:, None] * rv.cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        raise NotImplementedError

    def standard_normal(self, ndim, output_scale):
        raise NotImplementedError

    def update_mean(self, mean, x, /, num):
        sum_updated = _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x)
        return sum_updated / jnp.sqrt(num + 1)
