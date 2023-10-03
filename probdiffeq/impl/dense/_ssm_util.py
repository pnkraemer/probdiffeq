"""State-space model utilities."""

import jax.numpy as jnp

from probdiffeq.impl import _ssm_util
from probdiffeq.impl.dense import _normal
from probdiffeq.util import cholesky_util, cond_util, ibm_util


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives, output_scale):
        a, q_sqrtm = ibm_util.system_matrices_1d(num_derivatives, output_scale)
        (d,) = self.ode_shape
        eye_d = jnp.eye(d)
        A = jnp.kron(eye_d, a)
        Q = jnp.kron(eye_d, q_sqrtm)

        ndim = d * (num_derivatives + 1)
        q0 = jnp.zeros((ndim,))
        noise = _normal.Normal(q0, Q)

        precon_fun = ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            p = jnp.tile(p, d)
            p_inv = jnp.tile(p_inv, d)
            return cond_util.Conditional(A, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        (d,) = self.ode_shape
        n = ndim * d

        A = jnp.eye(n)
        m = jnp.zeros((n,))
        C = jnp.zeros((n, n))
        return cond_util.Conditional(A, _normal.Normal(m, C))

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = f"The number of Taylor coefficients {len(tcoeffs)} does not match "
            msg2 = f"the number of derivatives {num_derivatives+1} in the solver."
            raise ValueError(msg1 + msg2)

        if tcoeffs[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = jnp.stack(tcoeffs)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")

        (ode_dim,) = self.ode_shape
        ndim = (num_derivatives + 1) * ode_dim
        c_sqrtm0_corrected = jnp.zeros((ndim, ndim))

        return _normal.Normal(m0_corrected, c_sqrtm0_corrected)

    def preconditioner_apply(self, rv, p, /):
        mean = p * rv.mean
        cholesky = p[:, None] * rv.cholesky
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        noise = self.preconditioner_apply(noise, p)
        A = p[:, None] * A * p_inv[None, :]
        return cond_util.Conditional(A, noise)

    def standard_normal(self, ndim, /, output_scale):
        eye_n = jnp.eye(ndim)
        eye_d = output_scale * jnp.eye(*self.ode_shape)
        cholesky = jnp.kron(eye_d, eye_n)
        mean = jnp.zeros((*self.ode_shape, ndim)).reshape((-1,), order="F")
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        return cholesky_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x) / jnp.sqrt(
            num + 1
        )
