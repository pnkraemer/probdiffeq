"""Uncalibrated IVP solvers."""

from probdiffeq.ivpsolvers import _common
from probdiffeq.ivpsolvers import solver as solver_module


def solver(strategy, calibration_factory):
    """Create a solver that does not calibrate the output scale automatically."""
    string_repr = f"<Calibration-free solver with {strategy}>"
    return solver_module.Solver(
        strategy,
        calibration_factory.free(),
        string_repr=string_repr,
        step_fun=_step_calibration_free,
        requires_rescaling=False,
    )


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
