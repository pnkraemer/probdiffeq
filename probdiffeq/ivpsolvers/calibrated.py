"""Calibrated IVP solvers."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.ivpsolvers import _common, api


def mle(strategy, calibration_factory):
    """Create a solver that calibrates the output scale via maximum-likelihood."""
    string_repr = f"<MLE-solver with {strategy}>"
    return api.Solver(
        strategy,
        calibration_factory.mle(),
        string_repr=string_repr,
        step_fun=_step_mle,
        requires_rescaling=True,
    )


def _step_mle(state, /, dt, parameters, vector_field, *, strategy):
    state_strategy = strategy.begin(
        state.strategy,
        dt=dt,
        parameters=parameters,
        vector_field=vector_field,
    )
    (error, output_scale, _) = state_strategy.corr  # clean this up next?

    state_strategy = strategy.complete(
        state_strategy,
        output_scale=state.output_scale_prior,
        parameters=parameters,
        vector_field=vector_field,
    )
    observed = state_strategy.corr  # clean this up next?

    # Calibrate
    output_scale_calibrated = _mle_update_output_scale(
        state.output_scale_calibrated, observed, num_data=state.num_steps
    )
    return _common.State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale_prior=state.output_scale_prior,
        output_scale_calibrated=output_scale_calibrated,
        num_steps=state.num_steps + 1,
    )


def _mle_update_output_scale(output_scale_calibrated, observed, *, num_data):
    """Update the MLE of the output-scale."""
    # Special consideration for block-diagonal models:
    if jnp.ndim(output_scale_calibrated) > 0:
        # todo: move this function to calibration routines?
        fun_vmap = jax.vmap(lambda *a: _mle_update_output_scale(*a, num_data=num_data))
        return fun_vmap(output_scale_calibrated, observed)

    zero_data = jnp.zeros_like(observed.mean)
    mahalanobis_norm = observed.mahalanobis_norm(zero_data) / jnp.sqrt(zero_data.size)

    return _update_running_mean(output_scale_calibrated, mahalanobis_norm, num=num_data)


def _update_running_mean(x, y, /, num):
    sum_updated = _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * x, y)
    return sum_updated / jnp.sqrt(num + 1)


def dynamic(strategy, calibration_factory):
    """Create a solver that calibrates the output scale dynamically."""
    string_repr = f"<Dynamic solver with {strategy}>"
    return api.Solver(
        strategy,
        calibration_factory.dynamic(),
        string_repr=string_repr,
        step_fun=_step_dynamic,
        requires_rescaling=False,
    )


def _step_dynamic(state, /, dt, parameters, vector_field, *, strategy):
    state_strategy = strategy.begin(
        state.strategy,
        dt=dt,
        parameters=parameters,
        vector_field=vector_field,
    )
    (error, output_scale, _) = state_strategy.corr  # clean this up next?

    state_strategy = strategy.complete(
        state_strategy,
        parameters=parameters,
        vector_field=vector_field,
        output_scale=output_scale,
    )

    # Return solution
    return _common.State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale_calibrated=output_scale,
        # current scale becomes the new prior scale!
        #  this is because dynamic solvers assume a piecewise-constant model
        output_scale_prior=output_scale,
        num_steps=state.num_steps + 1,
    )
