import jax
import jax.numpy as jnp

from probdiffeq.impl import _hidden_model
from probdiffeq.impl.isotropic import _normal
from probdiffeq.impl.util import cholesky_util


class HiddenModelBackend(_hidden_model.HiddenModelBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self, rv):
        return rv.mean[..., 0, :]

    def marginal_nth_derivative(self, rv, i):
        if jnp.ndim(rv.mean) > 2:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        if i > jnp.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[i, :]
        cholesky = cholesky_util.triu_via_qr((rv.cholesky[i, :])[:, None].T).T
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[0, :]
