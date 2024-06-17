from probdiffeq.backend import abc, containers, functools
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array, Callable
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util, cond_util


class Transformation(containers.NamedTuple):
    matmul: Callable
    bias: Array


class TransformBackend(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, transformation, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, transformation, /):
        raise NotImplementedError


class ScalarTransform(TransformBackend):
    def marginalise(self, rv, transformation, /):
        # currently, assumes that A(rv.cholesky) is a vector, not a matrix.
        matmul, b = transformation
        cholesky_new = cholesky_util.triu_via_qr(matmul(rv.cholesky)[:, None])
        cholesky_new_squeezed = np.reshape(cholesky_new, ())
        return _normal.Normal(matmul(rv.mean) + b, cholesky_new_squeezed)

    def revert(self, rv, transformation, /):
        # Assumes that A maps a vector to a scalar...

        # Extract information
        A, b = transformation

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree
        #  to transformation_revert_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=A(rv.cholesky)[:, None], R_X=rv.cholesky.T
        )
        cholesky_obs = np.reshape(r_obs, ())
        cholesky_cor = r_cor.T
        gain = np.squeeze_along_axis(gain, axis=-1)

        # Gather terms and return
        m_cor = rv.mean - gain * (A(rv.mean) + b)
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(A(rv.mean) + b, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)


class DenseTransform(TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        cholesky_new = cholesky_util.triu_via_qr((A @ rv.cholesky).T).T
        return _normal.Normal(A @ rv.mean + b, cholesky_new)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree to
        #   revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky).T, R_X=cholesky.T
        )

        # Gather terms and return
        m_cor = mean - gain @ (A @ mean + b)
        corrected = _normal.Normal(m_cor, r_cor.T)
        observed = _normal.Normal(A @ mean + b, r_obs.T)
        return observed, cond_util.Conditional(gain, corrected)


class IsotropicTransform(TransformBackend):
    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky
        cholesky_new = cholesky_util.triu_via_qr((A @ cholesky).T)
        cholesky_squeezed = np.reshape(cholesky_new, ())
        return _normal.Normal((A @ mean) + b, cholesky_squeezed)

    def revert(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        # QR-decomposition
        # (todo: rename revert_conditional_noisefree
        #   to revert_transformation_cov_sqrt())
        r_obs, (r_cor, gain) = cholesky_util.revert_conditional_noisefree(
            R_X_F=(A @ cholesky).T, R_X=cholesky.T
        )
        cholesky_obs = np.reshape(r_obs, ())
        cholesky_cor = r_cor.T

        # Gather terms and return
        mean_observed = A @ mean + b
        m_cor = mean - gain * mean_observed
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)


class BlockDiagTransform(TransformBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def marginalise(self, rv, transformation, /):
        A, b = transformation
        mean, cholesky = rv.mean, rv.cholesky

        A_cholesky = A @ cholesky
        cholesky = functools.vmap(cholesky_util.triu_via_qr)(_transpose(A_cholesky))
        mean = A @ mean + b
        return _normal.Normal(mean, cholesky)

    def revert(self, rv, transformation, /):
        A, bias = transformation
        cholesky_upper = np.transpose(rv.cholesky, axes=(0, -1, -2))
        A_cholesky_upper = _transpose(A @ rv.cholesky)

        revert_fun = functools.vmap(cholesky_util.revert_conditional_noisefree)
        r_obs, (r_cor, gain) = revert_fun(A_cholesky_upper, cholesky_upper)
        cholesky_obs = _transpose(r_obs)
        cholesky_cor = _transpose(r_cor)

        # Gather terms and return
        mean_observed = (A @ rv.mean) + bias
        m_cor = rv.mean - (gain * (mean_observed[..., None]))[..., 0]
        corrected = _normal.Normal(m_cor, cholesky_cor)
        observed = _normal.Normal(mean_observed, cholesky_obs)
        return observed, cond_util.Conditional(gain, corrected)


def _transpose(arr, /):
    return np.transpose(arr, axes=(0, 2, 1))
