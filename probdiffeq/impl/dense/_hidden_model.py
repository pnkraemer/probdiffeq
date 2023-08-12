import jax
import jax.numpy as jnp

from probdiffeq.impl import _hidden_model
from probdiffeq.impl.dense import _normal
from probdiffeq.impl.util import cholesky_util


class HiddenModelBackend(_hidden_model.HiddenModelBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self, rv):
        if jnp.ndim(rv.mean) > 1:
            return jax.vmap(self.qoi)(rv)
        mean_reshaped = jnp.reshape(rv.mean, (-1, *self.ode_shape), order="F")
        return mean_reshaped[0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        m = self._select(rv.mean, i)
        c = jax.vmap(self._select, in_axes=(1, None), out_axes=1)(rv.cholesky, i)
        c = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(m, c.T)

    def _select(self, x, /, i):
        x_reshaped = jnp.reshape(x, (-1, *self.ode_shape), order="F")
        if i > x_reshaped.shape[0]:
            raise ValueError
        return x_reshaped[i]

    def qoi_from_sample(self, sample, /):
        sample_reshaped = jnp.reshape(sample, (-1, *self.ode_shape), order="F")
        return sample_reshaped[0]
