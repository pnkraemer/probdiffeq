import jax
import jax.numpy as jnp

from probdiffeq.impl import _hidden_model
from probdiffeq.impl.blockdiag import _normal
from probdiffeq.util import cholesky_util, cond_util, linop_util


class HiddenModelBackend(_hidden_model.HiddenModelBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self, rv):
        return rv.mean[..., 0]

    def marginal_nth_derivative(self, rv, i):
        if jnp.ndim(rv.mean) > 2:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        if i > jnp.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[:, i]
        cholesky = jax.vmap(cholesky_util.triu_via_qr)(
            (rv.cholesky[:, i, :])[..., None]
        )
        cholesky = jnp.transpose(cholesky, axes=(0, 2, 1))
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[..., 0]

    def conditional_to_derivative(self, i, standard_deviation):
        def A(x):
            return x[:, [i], ...]

        bias = jnp.zeros((*self.ode_shape, 1))
        eye = jnp.ones((*self.ode_shape, 1, 1)) * jnp.eye(1)[None, ...]
        noise = _normal.Normal(bias, standard_deviation * eye)
        linop = linop_util.parametrised_linop(lambda s, _p: A(s))
        return cond_util.Conditional(linop, noise)
