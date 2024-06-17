from probdiffeq.backend import abc, functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util, cond_util, linop_util


class HiddenModelBackend(abc.ABC):
    @abc.abstractmethod
    def qoi(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_from_sample(self, sample, /):
        raise NotImplementedError

    @abc.abstractmethod
    def conditional_to_derivative(self, i, standard_deviation):
        raise NotImplementedError


class DenseHiddenModel(HiddenModelBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self, rv):
        if np.ndim(rv.mean) > 1:
            return functools.vmap(self.qoi)(rv)
        mean_reshaped = np.reshape(rv.mean, (-1, *self.ode_shape), order="F")
        return mean_reshaped[0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        m = self._select(rv.mean, i)
        c = functools.vmap(self._select, in_axes=(1, None), out_axes=1)(rv.cholesky, i)
        c = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(m, c.T)

    def qoi_from_sample(self, sample, /):
        sample_reshaped = np.reshape(sample, (-1, *self.ode_shape), order="F")
        return sample_reshaped[0]

    # TODO: move to linearise.py?
    def conditional_to_derivative(self, i, standard_deviation):
        a0 = functools.partial(self._select, idx_or_slice=i)

        (d,) = self.ode_shape
        bias = np.zeros((d,))
        eye = np.eye(d)
        noise = _normal.Normal(bias, standard_deviation * eye)
        linop = linop_util.parametrised_linop(
            lambda s, _p: self._autobatch_linop(a0)(s)
        )
        return cond_util.Conditional(linop, noise)

    def _select(self, x, /, idx_or_slice):
        x_reshaped = np.reshape(x, (-1, *self.ode_shape), order="F")
        if isinstance(idx_or_slice, int) and idx_or_slice > x_reshaped.shape[0]:
            raise ValueError
        return x_reshaped[idx_or_slice]

    @staticmethod
    def _autobatch_linop(fun):
        def fun_(x):
            if np.ndim(x) > 1:
                return functools.vmap(fun_, in_axes=1, out_axes=1)(x)
            return fun(x)

        return fun_


class IsotropicHiddenModel(HiddenModelBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self, rv):
        return rv.mean[..., 0, :]

    def marginal_nth_derivative(self, rv, i):
        if np.ndim(rv.mean) > 2:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        if i > np.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[i, :]
        cholesky = cholesky_util.triu_via_qr(rv.cholesky[[i], :].T).T
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[0, :]

    def conditional_to_derivative(self, i, standard_deviation):
        def A(x):
            return x[[i], ...]

        bias = np.zeros(self.ode_shape)
        eye = np.eye(1)
        noise = _normal.Normal(bias, standard_deviation * eye)
        linop = linop_util.parametrised_linop(lambda s, _p: A(s))
        return cond_util.Conditional(linop, noise)
