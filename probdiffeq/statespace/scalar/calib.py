"""Calibration."""
import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    return CalibrationFactory()


class ScalarMostRecent(_calib.Calibration):
    def init(self, prior):
        return prior

    def update(self, _state, /, observed):
        zero_data = jnp.zeros_like(observed.mean)
        mahalanobis_norm = observed.mahalanobis_norm(zero_data)
        calibrated = mahalanobis_norm / jnp.sqrt(zero_data.size)
        return calibrated

    def extract(self, state, /):
        return state, state


class CalibrationFactory(_calib.CalibrationFactory):
    pass


# todo: run tests for scalar solvers
