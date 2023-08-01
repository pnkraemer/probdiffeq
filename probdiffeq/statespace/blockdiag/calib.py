"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale(output_scale_scalar, *, ode_shape):
    """Construct (a buffet of) isotropic calibration strategies."""
    return _BlockDiagCalibrationFactory(output_scale_scalar, ode_shape=ode_shape)


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
