"""Calibrated IVP solvers."""

import jax

from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.ivpsolvers import _common, solver


def mle(strategy, calibration_factory):
    """Create a solver that calibrates the output scale via maximum-likelihood."""
    string_repr = f"<MLE-solver with {strategy}>"
    return CalibratedSolver(
        calibration=calibration_factory.running_mean(),
        impl_step=_step_mle,
        strategy=strategy,
        string_repr=string_repr,
        requires_rescaling=True,
    )


def _step_mle(state, /, dt, parameters, vector_field, *, strategy, calibration):
    output_scale_prior, _calibrated = calibration.extract(state.output_scale)
    error, _, state_strategy = strategy.predict_error(
        state.strategy,
        dt=dt,
        parameters=parameters,
        vector_field=vector_field,
    )

    state_strategy = strategy.complete(state_strategy, output_scale=output_scale_prior)
    observed = state_strategy.aux_corr

    # Calibrate
    output_scale = calibration.update(state.output_scale, observed=observed)

    # Return
    return _common.State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale=output_scale,
        num_steps=state.num_steps + 1,
    )


def dynamic(strategy, calibration_factory):
    """Create a solver that calibrates the output scale dynamically."""
    string_repr = f"<Dynamic solver with {strategy}>"
    return CalibratedSolver(
        strategy=strategy,
        calibration=calibration_factory.most_recent(),
        string_repr=string_repr,
        impl_step=_step_dynamic,
        requires_rescaling=False,
    )


def _step_dynamic(state, /, dt, parameters, vector_field, *, strategy, calibration):
    error, observed, state_strategy = strategy.predict_error(
        state.strategy,
        dt=dt,
        parameters=parameters,
        vector_field=vector_field,
    )

    output_scale = calibration.update(state.output_scale, observed=observed)

    prior, _calibrated = calibration.extract(output_scale)
    state_strategy = strategy.complete(state_strategy, output_scale=prior)

    # Return solution
    return _common.State(
        error_estimate=dt * error,
        strategy=state_strategy,
        output_scale=output_scale,
        num_steps=state.num_steps + 1,
    )


class CalibratedSolver(solver.Solver[_common.State]):
    def __init__(self, *, calibration, impl_step, **kwargs):
        super().__init__(**kwargs)

        self.calibration = calibration
        self.impl_step = impl_step

    def init(self, t, posterior, /, output_scale, num_steps) -> _common.State:
        state_strategy = self.strategy.init(t, posterior)
        error_estimate = impl.ssm_util.prototype_error_estimate()
        calib_state = self.calibration.init(output_scale)
        return _common.State(
            error_estimate=error_estimate,
            strategy=state_strategy,
            output_scale=calib_state,
            num_steps=num_steps,
        )

    def step(
        self, state: _common.State, *, vector_field, dt, parameters
    ) -> _common.State:
        return self.impl_step(
            state,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
            strategy=self.strategy,
            calibration=self.calibration,
        )

    def extract(self, state: _common.State, /):
        t, posterior = self.strategy.extract(state.strategy)
        _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
        return t, posterior, output_scale, state.num_steps

    def interpolate(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        output_scale, _ = self.calibration.extract(s1.output_scale)
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            t, s0=s0.strategy, s1=s1.strategy, output_scale=output_scale
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def right_corner(self, state_at_t0, state_at_t1) -> _interp.InterpRes:
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(state_at_t1.strategy)

        prev = self._interp_make_state(prev_p, reference=state_at_t0)
        sol = self._interp_make_state(sol_p, reference=state_at_t1)
        acc = self._interp_make_state(acc_p, reference=state_at_t1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def _interp_make_state(
        self, state_strategy, *, reference: _common.State
    ) -> _common.State:
        error_estimate = impl.ssm_util.prototype_error_estimate()
        return _common.State(
            strategy=state_strategy,
            error_estimate=error_estimate,
            output_scale=reference.output_scale,
            num_steps=reference.num_steps,
        )


def _solver_flatten(solver):
    children = (solver.strategy, solver.calibration)
    aux = (solver.impl_step, solver.requires_rescaling, solver.string_repr)
    return children, aux


def _solver_unflatten(aux, children):
    strategy, calibration = children
    impl_step, rescaling, string_repr = aux
    return CalibratedSolver(
        strategy=strategy,
        calibration=calibration,
        impl_step=impl_step,
        requires_rescaling=rescaling,
        string_repr=string_repr,
    )


jax.tree_util.register_pytree_node(CalibratedSolver, _solver_flatten, _solver_unflatten)
