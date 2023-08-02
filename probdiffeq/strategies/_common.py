"""Common functionalities among all strategies."""

from typing import Any

from probdiffeq.backend import containers


class State(containers.NamedTuple):
    t: Any
    ssv: Any
    extra: Any

    corr: Any

    @property
    def u(self):
        return self.ssv.u
