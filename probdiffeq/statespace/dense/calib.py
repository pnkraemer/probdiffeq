"""Calibration tools."""

import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    return DenseFactory()


class DenseMostRecent(_calib.Calibration):
    def init(self, prior):
        return prior

    def update(self, state, /, observed):
        zero_data = jnp.zeros_like(observed.mean)
        mahalanobis_norm = observed.mahalanobis_norm(zero_data)
        calibrated = mahalanobis_norm / jnp.sqrt(zero_data.size)
        return calibrated

    def extract(self, state, /):
        return state, state


class DenseRunningMean(_calib.Calibration):
    def init(self, prior):
        raise NotImplementedError

    def update(self, state, /, observed):
        raise NotImplementedError

    def extract(self, state, /):
        raise NotImplementedError


class DenseFactory(_calib.CalibrationFactory):
    def most_recent(self) -> DenseMostRecent:
        return DenseMostRecent()

    def running_mean(self) -> DenseRunningMean:
        return DenseRunningMean()
