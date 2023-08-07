import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _ibm_util, _ssm_util, matfree
from probdiffeq.impl.isotropic import _normal


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives, output_scale=1.0)
        q0 = jnp.zeros((num_derivatives + 1,) + self.ode_shape)
        noise = _normal.Normal(q0, q_sqrtm)
        A = matfree.linop_from_matmul(a)
        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (A, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, num_hidden_states_per_ode_dim, /):
        m0 = jnp.zeros((num_hidden_states_per_ode_dim,) + self.ode_shape)
        c0 = jnp.zeros((num_hidden_states_per_ode_dim, num_hidden_states_per_ode_dim))
        noise = _normal.Normal(m0, c0)
        matrix = jnp.eye(num_hidden_states_per_ode_dim)
        return matrix, noise

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
        A, noise = cond
        A = p[:, None] * A * p_inv[None, :]
        noise = _normal.Normal(p[:, None] * noise.mean, p[:, None] * noise.cholesky)
        return A, noise

    def standard_normal(self, num, /, output_scale):
        mean = jnp.zeros((num,) + self.ode_shape)
        cholesky = output_scale * jnp.eye(num)
        return _normal.Normal(mean, cholesky)

    def update_mean(self, mean, x, /, num):
        sum_updated = _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x)
        return sum_updated / jnp.sqrt(num + 1)

    def conditional_to_derivative(self, i, standard_deviation):
        def A(x):
            return x[i, ...]

        bias = jnp.zeros(self.ode_shape)
        eye = jnp.eye(*self.ode_shape)
        return A, _normal.Normal(bias, standard_deviation * eye)
