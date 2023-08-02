from typing import Any

from probdiffeq.backend import containers


class State(containers.NamedTuple):
    """Solver state."""

    strategy: Any

    error_estimate: Any
    output_scale_calibrated: Any
    output_scale_prior: Any

    num_steps: Any

    @property
    def t(self):
        return self.strategy.t

    @property
    def u(self):
        return self.strategy.u
