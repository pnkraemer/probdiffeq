"""Calibration API."""


from typing import Callable

from probdiffeq.backend import containers


class Calib(containers.NamedTuple):
    # todo: no "apply" method yet. Maybe in the future.

    init: Callable
    update: Callable
    extract: Callable


class CalibrationBundle(containers.NamedTuple):
    mle: Calib
    dynamic: Calib
    free: Calib
