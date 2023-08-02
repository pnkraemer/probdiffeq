"""Calibrated IVP solvers."""

from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _interp, _sqrt_util, solution
from probdiffeq.backend import containers


def solver_mle(strategy, calibration_factory):
    """Create a solver that calibrates the output scale via maximum-likelihood."""
    string_repr = f"<MLE-solver with {strategy}>"
    return Solver(
        strategy,
        calibration_factory.mle(),
        string_repr=string_repr,
        step_fun=_step_mle,
        requires_rescaling=True,
    )


def solver_dynamic(strategy, calibration_factory):
    """Create a solver that calibrates the output scale dynamically."""
    string_repr = f"<Dynamic solver with {strategy}>"
    return Solver(
        strategy,
        calibration_factory.dynamic(),
        string_repr=string_repr,
        step_fun=_step_dynamic,
        requires_rescaling=False,
    )


def solver_calibrationfree(strategy, calibration_factory):
    """Create a solver that does not calibrate the output scale automatically."""
    string_repr = f"<Calibration-free solver with {strategy}>"
    return Solver(
        strategy,
        calibration_factory.free(),
        string_repr=string_repr,
        step_fun=_step_calibration_free,
        requires_rescaling=False,
    )


class _State(containers.NamedTuple):
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


@jax.tree_util.register_pytree_node_class
class Solver:
    """Interface for initial value problem solvers."""

    def __init__(
        self, strategy, calibration, /, *, step_fun, string_repr, requires_rescaling
    ):
        self.strategy = strategy
        self.calibration = calibration

        self.requires_rescaling = requires_rescaling

        self._step_fun = step_fun
        self._string_repr = string_repr

    def __repr__(self):
        return self._string_repr

    def solution_from_tcoeffs(self, tcoeffs, /, t, output_scale, num_steps=1.0):
        """Construct an initial `Solution` object.

        An (even if empty) solution object is needed to initialise the solver.
        Thus, this method is kind-of a helper function to make the rest of the
        initialisation code a bit simpler.
        """
        u, marginals, posterior = self.strategy.solution_from_tcoeffs(tcoeffs)
        return solution.Solution(
            t=t,
            posterior=posterior,
            marginals=marginals,
            output_scale=output_scale,
            u=u,
            num_steps=num_steps,
        )

    def init(self, t, posterior, /, output_scale, num_steps) -> _State:
        state_strategy = self.strategy.init(t, posterior)
        error_estimate = jnp.empty_like(state_strategy.u)
        output_scale = self.calibration.init(output_scale)
        return _State(
            error_estimate=error_estimate,
            strategy=state_strategy,
            output_scale_prior=output_scale,
            output_scale_calibrated=output_scale,
            num_steps=num_steps,
        )

    def step(self, *, state: _State, vector_field, dt, parameters) -> _State:
        return self._step_fun(
            state, dt, parameters, vector_field, strategy=self.strategy
        )

    def extract(self, state: _State, /):
        t, posterior = self.strategy.extract(state.strategy)
        output_scale = self.calibration.extract(state.output_scale_prior)
        return t, posterior, output_scale, state.num_steps

    def interpolate_fun(self, t, s0: _State, s1: _State) -> _interp.InterpRes[_State]:
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            # always interpolate with the prior output scale.
            #  This is important to make the MLE solver behave correctly.
            #  (Dynamic solvers overwrite the prior output scale at every step anyway).
            output_scale=s1.output_scale_prior,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def right_corner_fun(self, t, s0: _State, s1: _State) -> _interp.InterpRes[_State]:
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            output_scale=s1.output_scale_prior,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def _interp_make_state(self, state_strategy, *, reference: _State) -> _State:
        error_estimate = jnp.empty_like(state_strategy.u)
        return _State(
            strategy=state_strategy,
            error_estimate=error_estimate,
            output_scale_prior=reference.output_scale_prior,
            output_scale_calibrated=reference.output_scale_calibrated,
            num_steps=reference.num_steps,
        )

    def tree_flatten(self):
        children = (self.strategy, self.calibration)
        aux = (self._step_fun, self.requires_rescaling, self._string_repr)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        strategy, calibration = children
        step_fun, rescaling, string_repr = aux
        return cls(
            strategy,
            calibration,
            step_fun=step_fun,
            requires_rescaling=rescaling,
            string_repr=string_repr,
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
    return _State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale_prior=state.output_scale_prior,
        # Nothing happens in the field below:
        #  but we cannot use "None" if we want to reuse the init()
        #  method from abstract solvers (which populate this field).
        output_scale_calibrated=state.output_scale_prior,
        num_steps=state.num_steps + 1,
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
    return _State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale_calibrated=output_scale,
        # current scale becomes the new prior scale!
        #  this is because dynamic solvers assume a piecewise-constant model
        output_scale_prior=output_scale,
        num_steps=state.num_steps + 1,
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
    return _State(
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
