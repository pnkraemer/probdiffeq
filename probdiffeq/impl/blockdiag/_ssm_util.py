"""State-space model utilities."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _cond_util, _ibm_util, _matfree, _sqrt_util, _ssm_util
from probdiffeq.impl.blockdiag import _normal


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ibm_transitions(self, num_derivatives):
        output_scale = jnp.ones(self.ode_shape)
        system_matrices = jax.vmap(_ibm_util.system_matrices_1d, in_axes=(None, 0))
        a, q_sqrtm = system_matrices(num_derivatives, output_scale)

        q0 = jnp.zeros((*self.ode_shape, num_derivatives + 1))
        noise = _normal.Normal(q0, q_sqrtm)

        precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

        def discretise(dt):
            p, p_inv = precon_fun(dt)
            return (a, noise), (p, p_inv)

        return discretise

    def identity_conditional(self, ndim, /):
        m0 = jnp.zeros((*self.ode_shape, ndim))
        c0 = jnp.zeros((*self.ode_shape, ndim, ndim))
        noise = _normal.Normal(m0, c0)

        matrix = jnp.ones((*self.ode_shape, 1, 1)) * jnp.eye(ndim, ndim)[None, ...]
        return _cond_util.Conditional(matrix, noise)

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        cholesky_shape = (*self.ode_shape, num_derivatives + 1, num_derivatives + 1)
        cholesky = jnp.zeros(cholesky_shape)
        mean = jnp.stack(tcoeffs).T
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply(self, rv, p, /):
        mean = p[None, :] * rv.mean
        cholesky = p[None, :, None] * rv.cholesky
        return _normal.Normal(mean, cholesky)

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        A, noise = cond
        A_new = p[None, :, None] * A * p_inv[None, None, :]
        noise = self.preconditioner_apply(noise, p)
        return _cond_util.Conditional(A_new, noise)

    def standard_normal(self, ndim, output_scale):
        raise NotImplementedError

    def update_mean(self, mean, x, /, num):
        if jnp.ndim(mean) > 0:
            assert jnp.shape(mean) == jnp.shape(x)
            return jax.vmap(self.update_mean, in_axes=(0, 0, None))(mean, x, num)

        sum_updated = _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x)
        return sum_updated / jnp.sqrt(num + 1)

    def conditional_to_derivative(self, i, standard_deviation):
        def A(x):
            return x[:, [i], ...]

        bias = jnp.zeros((*self.ode_shape, 1))
        eye = jnp.ones((*self.ode_shape, 1, 1)) * jnp.eye(1)[None, ...]
        noise = _normal.Normal(bias, standard_deviation * eye)
        linop = _matfree.parametrised_linop(lambda s, _p: A(s))
        return _cond_util.Conditional(linop, noise)

    def prototype_qoi(self):
        mean = jnp.empty((*self.ode_shape, 1))
        cholesky = jnp.empty((*self.ode_shape, 1, 1))
        return _normal.Normal(mean, cholesky)

    def prototype_error_estimate(self):
        return jnp.empty(self.ode_shape)

    def prototype_output_scale(self):
        return jnp.empty(self.ode_shape)
