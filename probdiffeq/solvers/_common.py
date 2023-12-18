from probdiffeq.backend import containers
from probdiffeq.backend.typing import Any


class State(containers.NamedTuple):
    """Solver state."""

    strategy: Any
    output_scale: Any

    @property
    def t(self):
        return self.strategy.t
