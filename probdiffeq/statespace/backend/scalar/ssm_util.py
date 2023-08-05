"""SSM utilities."""
import jax.numpy as jnp

from probdiffeq.statespace import _ibm_util
from probdiffeq.statespace.backend import _ssm_util
from probdiffeq.statespace.backend.scalar import random


class SSMUtilBackEnd(_ssm_util.SSMUtilBackEnd):
    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(tcoeffs)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
        return random.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        return random.Normal(p * rv.mean, p[:, None] * rv.cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        A = p[:, None] * A * p_inv[None, :]
        noise = random.Normal(p * noise.mean, p[:, None] * noise.cholesky)
        return A, noise

    def ibm_transitions(self, num_derivatives, output_scale):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives, output_scale)
        q0 = jnp.zeros((num_derivatives + 1,))
        noise = random.Normal(q0, q_sqrtm)

        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim):
        transition = jnp.eye(ndim)

        mean = jnp.zeros((ndim,))
        cov_sqrtm = jnp.zeros((ndim, ndim))
        noise = random.Normal(mean, cov_sqrtm)
        return transition, noise

    def standard_normal(self, ndim, output_scale):
        mean = jnp.zeros((ndim,))
        cholesky = output_scale * jnp.eye(ndim)
        return random.Normal(mean, cholesky)
