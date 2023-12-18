"""Calibrated IVP solvers."""
from probdiffeq import _interp
from probdiffeq.backend import abc, tree_util
from probdiffeq.impl import impl
from probdiffeq.solvers import _common, _solver


def mle(strategy):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to solution.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """
    string_repr = f"<MLE-solver with {strategy}>"
    return _CalibratedSolver(
        calibration=_RunningMean(),
        impl_step=_step_mle,
        strategy=strategy,
        string_repr=string_repr,
        requires_rescaling=True,
    )


def _step_mle(state, /, dt, vector_field, *, strategy, calibration):
    output_scale_prior, _calibrated = calibration.extract(state.output_scale)
    error, _, state_strategy = strategy.predict_error(
        state.strategy, dt=dt, vector_field=vector_field
    )

    state_strategy = strategy.complete(state_strategy, output_scale=output_scale_prior)
    observed = state_strategy.aux_corr

    # Calibrate
    output_scale = calibration.update(state.output_scale, observed=observed)

    # Return
    state = _common.State(strategy=state_strategy, output_scale=output_scale)
    return dt * error, state


def dynamic(strategy):
    """Create a solver that calibrates the output scale dynamically."""
    string_repr = f"<Dynamic solver with {strategy}>"
    return _CalibratedSolver(
        strategy=strategy,
        calibration=_MostRecent(),
        string_repr=string_repr,
        impl_step=_step_dynamic,
        requires_rescaling=False,
    )


def _step_dynamic(state, /, dt, vector_field, *, strategy, calibration):
    error, observed, state_strategy = strategy.predict_error(
        state.strategy, dt=dt, vector_field=vector_field
    )

    output_scale = calibration.update(state.output_scale, observed=observed)

    prior, _calibrated = calibration.extract(output_scale)
    state_strategy = strategy.complete(state_strategy, output_scale=prior)

    # Return solution
    state = _common.State(strategy=state_strategy, output_scale=output_scale)
    return dt * error, state


class _Calibration(abc.ABC):
    """Calibration implementation."""

    @abc.abstractmethod
    def init(self, prior):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, /, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state, /):
        raise NotImplementedError


class _MostRecent(_Calibration):
    def init(self, prior):
        return prior

    def update(self, _state, /, observed):
        return impl.stats.mahalanobis_norm_relative(0.0, observed)

    def extract(self, state, /):
        return state, state


# TODO: if we pass the mahalanobis_relative term to the update() function,
#  it reduces to a generic stats() module that can also be used for e.g.
#  marginal likelihoods. In this case, the _MostRecent() stuff becomes void.
class _RunningMean(_Calibration):
    def init(self, prior):
        return prior, prior, 0.0

    def update(self, state, /, observed):
        prior, calibrated, num_data = state

        new_term = impl.stats.mahalanobis_norm_relative(0.0, observed)
        calibrated = impl.ssm_util.update_mean(calibrated, new_term, num=num_data)
        return prior, calibrated, num_data + 1.0

    def extract(self, state, /):
        prior, calibrated, _num_data = state
        return prior, calibrated


def _unflatten_func(nodetype):
    return lambda *_a: nodetype()


# Register objects as (empty) pytrees. todo: temporary?!
for node in [_RunningMean, _MostRecent]:
    tree_util.register_pytree_node(
        node, flatten_func=lambda _: ((), ()), unflatten_func=_unflatten_func(node)
    )


class _CalibratedSolver(_solver.Solver):
    def __init__(self, *, calibration: _Calibration, impl_step, **kwargs):
        super().__init__(**kwargs)

        self.calibration = calibration
        self.impl_step = impl_step

    def init(self, t, initial_condition) -> _common.State:
        posterior, output_scale = initial_condition
        state_strategy = self.strategy.init(t, posterior)
        calib_state = self.calibration.init(output_scale)
        return _common.State(strategy=state_strategy, output_scale=calib_state)

    def step(self, state: _common.State, *, vector_field, dt) -> _common.State:
        return self.impl_step(
            state,
            vector_field=vector_field,
            dt=dt,
            strategy=self.strategy,
            calibration=self.calibration,
        )

    def extract(self, state: _common.State, /):
        t, posterior = self.strategy.extract(state.strategy)
        _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
        return t, (posterior, output_scale)

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
        return _common.State(state_strategy, output_scale=reference.output_scale)


def _slvr_flatten(solver):
    children = (solver.strategy, solver.calibration)
    aux = (solver.impl_step, solver.requires_rescaling, solver.string_repr)
    return children, aux


def _slvr_unflatten(aux, children):
    strategy, calibration = children
    impl_step, rescaling, string_repr = aux
    return _CalibratedSolver(
        strategy=strategy,
        calibration=calibration,
        impl_step=impl_step,
        requires_rescaling=rescaling,
        string_repr=string_repr,
    )


tree_util.register_pytree_node(_CalibratedSolver, _slvr_flatten, _slvr_unflatten)
