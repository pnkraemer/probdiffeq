"""State-space model utilities."""
import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.impl import _cond_util, _ibm_util, _ssm_util
from probdiffeq.impl.blockdiag import _normal


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives):
        output_scale = jnp.ones(self.ode_shape)
        system_matrices = jax.vmap(_ibm_util.system_matrices_1d, in_axes=(None, 0))
        a, q_sqrtm = system_matrices(num_derivatives, output_scale)

        q0 = jnp.zeros(self.ode_shape + (num_derivatives + 1,))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        matrix_shape = self.ode_shape + (ndim, ndim)
        m0 = jnp.zeros(self.ode_shape + (ndim,) + self.ode_shape)
        c0 = jnp.zeros(matrix_shape)
        noise = _normal.Normal(m0, c0)

        matrix = jnp.eye(*matrix_shape)
        return _cond_util.Conditional(matrix, noise)

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        cholesky_shape = self.ode_shape + (num_derivatives + 1, num_derivatives + 1)
        cholesky = jnp.zeros(cholesky_shape)
        mean = jnp.stack(tcoeffs).T
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        raise NotImplementedError

    def standard_normal(self, ndim, output_scale):
        raise NotImplementedError

    def update_mean(self, mean, x, /, num):
        print(mean.shape, x.shape)

        sum_of_scalars = jax.vmap(_sqrt_util.sqrt_sum_square_scalar)
        sum_updated = sum_of_scalars(jnp.sqrt(num) * mean, x)
        return sum_updated / jnp.sqrt(num + 1)

    def conditional_to_derivative(self, i, standard_deviation):
        raise NotImplementedError
