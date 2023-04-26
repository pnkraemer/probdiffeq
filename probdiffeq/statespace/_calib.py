"""Calibration API."""


from typing import Any

from probdiffeq.backend import containers


class Calib(containers.NamedTuple):
    # todo: no "apply" method yet. Maybe in the future.

    init: Any
    extract: Any
