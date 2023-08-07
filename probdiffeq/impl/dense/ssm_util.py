import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _ibm_util, _ssm_util
from probdiffeq.impl.dense import _normal


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives, output_scale=1.0)
        (d,) = self.ode_shape
        eye_d = jnp.eye(d)
        A = jnp.kron(eye_d, a)
        Q = jnp.kron(eye_d, q_sqrtm)

        ndim = d * (num_derivatives + 1)
        q0 = jnp.zeros((ndim,))
        noise = _normal.Normal(q0, Q)

        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            p = jnp.tile(p, d)
            p_inv = jnp.tile(p_inv, d)
            return (A, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, num_hidden_states_per_ode_dim):
        raise NotImplementedError

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
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
        raise NotImplementedError

    def standard_normal(self, num_hidden_states_per_ode_dim, output_scale):
        raise NotImplementedError

    def update_mean(self, mean, x, /, num):
        return _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x)

    def conditional_to_derivative(self, i, standard_deviation):
        raise NotImplementedError
