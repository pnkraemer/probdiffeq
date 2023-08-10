"""Common functionalities among all strategies."""

from typing import Any

from probdiffeq.backend import containers


class State(containers.NamedTuple):
    t: Any
    hidden: Any
    aux_extra: Any
    aux_corr: Any
