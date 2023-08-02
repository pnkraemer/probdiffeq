"""Uncalibrated IVP solvers."""

from probdiffeq import _interp
from probdiffeq.ivpsolvers import _common
from probdiffeq.ivpsolvers import solver as solver_module


def solver(strategy):
    """Create a solver that does not calibrate the output scale automatically."""
    string_repr = f"<Uncalibrated solver with {strategy}>"
    return _Solver(string_repr=string_repr, requires_rescaling=False)


class _Solver(solver_module.Solver[_common.State]):
    def init(self, t, posterior, /, output_scale, num_steps) -> _common.State:
        state_strategy = self.strategy.init(t, posterior)
        error_estimate = jnp.empty_like(state_strategy.u)
        return _common.State(
            error_estimate=error_estimate,
            strategy=state_strategy,
            output_scale=output_scale,
            num_steps=num_steps,
        )

    def step(
        self, state: _common.State, *, vector_field, dt, parameters
    ) -> _common.State:
        raise NotImplementedError

    def extract(self, state: _common.State, /):
        raise NotImplementedError

    def interpolate(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        raise NotImplementedError

    def right_corner(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        raise NotImplementedError


def _step_calibration_free(state, /, dt, parameters, vector_field, *, strategy):
    state_strategy = strategy.begin(
        state.strategy,
        dt=dt,
        parameters=parameters,
        vector_field=vector_field,
    )
    (error, _, cache_obs) = state_strategy.corr
    state_strategy = strategy.complete(
        state_strategy,
        parameters=parameters,
        vector_field=vector_field,
        output_scale=state.output_scale_prior,
    )
    # Extract and return solution
    return _common.State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale_prior=state.output_scale_prior,
        # Nothing happens in the field below:
        #  but we cannot use "None" if we want to reuse the init()
        #  method from abstract solvers (which populate this field).
        output_scale_calibrated=state.output_scale_prior,
        num_steps=state.num_steps + 1,
    )
