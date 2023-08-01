"""Calibration tools."""

import jax

from probdiffeq.statespace import _calib


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    return _DenseCalibrationFactory()


class _DenseCalibrationFactory(_calib.CalibrationFactory):
    def dynamic(self) -> _calib.Calibration:
        return _output_scale_dynamic()

    def mle(self) -> _calib.Calibration:
        return _output_scale_mle()

    def free(self) -> _calib.Calibration:
        return _output_scale_free()


def _output_scale_mle():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):
        return s

    @jax.tree_util.Partial
    def extract(s, /):
        return s

    return _calib.Calibration(init=init, update=update, extract=extract)


def _output_scale_dynamic():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):  # todo: make correct
        return s

    @jax.tree_util.Partial
    def extract(s):
        return s

    return _calib.Calibration(init=init, update=update, extract=extract)


def _output_scale_free():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def update(s, /):
        return s

    @jax.tree_util.Partial
    def extract(s, /):
        return s

    return _calib.Calibration(init=init, update=update, extract=extract)
