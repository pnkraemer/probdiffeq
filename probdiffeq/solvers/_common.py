from typing import Any

from probdiffeq.backend import containers


class State(containers.NamedTuple):
    """Solver state."""

    strategy: Any
    output_scale: Any

    @property
    def t(self):
        return self.strategy.t
