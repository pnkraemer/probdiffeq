"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq.statespace import _calib
from probdiffeq.statespace.scalar import calib


def output_scale(ode_shape):
    """Construct (a buffet of) isotropic calibration strategies."""
    return BlockDiagFactory(ode_shape=ode_shape)


class BlockDiagMostRecent(_calib.Calibration):
    def __init__(self, *, ode_shape):
        self.ode_shape = ode_shape

        self.wraps = calib.ScalarMostRecent()

    def init(self, prior):
        prior_promoted = prior * jnp.ones(self.ode_shape)
        return jax.vmap(self.wraps.init)(prior_promoted)

    def update(self, state, /, observed):
        return jax.vmap(self.wraps.update)(state, observed)

    def extract(self, state, /):
        return jax.vmap(self.wraps.extract)(state)


class BlockDiagRunningMean(_calib.Calibration):
    def __init__(self, *, ode_shape):
        self.ode_shape = ode_shape

    def init(self, prior):
        raise NotImplementedError

    def update(self, state, /, observed):
        raise NotImplementedError

    def extract(self, state, /):
        raise NotImplementedError


class BlockDiagFactory(_calib.CalibrationFactory):
    def __init__(self, *, ode_shape):
        self.ode_shape = ode_shape

    def most_recent(self) -> BlockDiagMostRecent:
        return BlockDiagMostRecent(ode_shape=self.ode_shape)

    def running_mean(self) -> BlockDiagRunningMean:
        return BlockDiagRunningMean(ode_shape=self.ode_shape)


class _BlockDiagCalibrationFactory(_calib.CalibrationFactory):
    def __init__(self, wraps, ode_shape):
        self.wraps = wraps
        self.ode_shape = ode_shape

    def dynamic(self) -> _calib.Calibration:
        return _blockdiag(self.wraps.dynamic(), ode_shape=self.ode_shape)

    def mle(self) -> _calib.Calibration:
        return _blockdiag(self.wraps.mle(), ode_shape=self.ode_shape)

    def free(self) -> _calib.Calibration:
        return _blockdiag(self.wraps.free(), ode_shape=self.ode_shape)


def _blockdiag(output_scale_scalar, *, ode_shape):
    @jax.tree_util.Partial
    def init(s, /):
        # todo: raise warning/error if shape has to be promoted?
        #  don't just promote by default.
        s_arr = s * jnp.ones(ode_shape)
        return jax.vmap(output_scale_scalar.init)(s_arr)

    @jax.tree_util.Partial
    def update(s, /):
        return jax.vmap(output_scale_scalar.update)(s)

    @jax.tree_util.Partial
    def extract(s):
        if jnp.ndim(s) > 1:
            # For block-diagonal SSMs:
            # The shape of the solution is (N, d, ...). So we have to vmap along axis=1
            # This only affects extract() because the other functions assume (N,) = ().
            return jax.vmap(output_scale_scalar.extract, in_axes=1, out_axes=1)(s)

        return jax.vmap(output_scale_scalar.extract)(s)

    return _calib.Calibration(init=init, update=update, extract=extract)
