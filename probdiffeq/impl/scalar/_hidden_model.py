"""Hidden state-space model implementation."""
from probdiffeq.backend import functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _hidden_model
from probdiffeq.impl.scalar import _normal
from probdiffeq.util import cholesky_util, cond_util, linop_util


class HiddenModelBackend(_hidden_model.HiddenModelBackend):
    def qoi(self, rv):
        return rv.mean[..., 0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        if i > rv.mean.shape[0]:
            raise ValueError

        m = rv.mean[i]
        c = rv.cholesky[[i], :]
        chol = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(np.reshape(m, ()), np.reshape(chol, ()))

    def qoi_from_sample(self, sample, /):
        return sample[0]

    def conditional_to_derivative(self, i, standard_deviation):
        def A(x):
            return x[[i], ...]

        bias = np.zeros(())
        eye = np.eye(1)
        noise = _normal.Normal(bias, standard_deviation * eye)
        linop = linop_util.parametrised_linop(lambda s, _p: A(s))
        return cond_util.Conditional(linop, noise)
