from typing import Any

from probdiffeq.backend import containers


class State(containers.NamedTuple):
    """Solver state."""

    strategy: Any

    error_estimate: Any
    output_scale: Any

    num_steps: Any

    @property
    def t(self):
        return self.strategy.t
