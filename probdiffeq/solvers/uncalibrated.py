"""Uncalibrated IVP solvers."""

from probdiffeq import _interp
from probdiffeq.backend import tree_util
from probdiffeq.solvers import _common, _solver


def solver(strategy, /):
    """Create a solver that does not calibrate the output scale automatically."""
    string_repr = f"<Uncalibrated solver with {strategy}>"
    return _UncalibratedSolver(
        strategy=strategy, string_repr=string_repr, requires_rescaling=False
    )


class _UncalibratedSolver(_solver.Solver[_common.State]):
    def init(self, t, initial_condition) -> _common.State:
        posterior, output_scale = initial_condition
        state_strategy = self.strategy.init(t, posterior)
        return _common.State(strategy=state_strategy, output_scale=output_scale)

    def step(self, state: _common.State, *, vector_field, dt):
        error, _observed, state_strategy = self.strategy.predict_error(
            state.strategy, dt=dt, vector_field=vector_field
        )
        state_strategy = self.strategy.complete(
            state_strategy, output_scale=state.output_scale
        )
        # Extract and return solution
        state = _common.State(strategy=state_strategy, output_scale=state.output_scale)
        return dt * error, state

    def extract(self, state: _common.State, /):
        t, posterior = self.strategy.extract(state.strategy)
        return t, (posterior, state.output_scale)

    def interpolate(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            t, s0=s0.strategy, s1=s1.strategy, output_scale=s1.output_scale
        )
        prev = _common.State(prev_p, output_scale=s0.output_scale)
        sol = _common.State(sol_p, output_scale=s1.output_scale)
        acc = _common.State(acc_p, output_scale=s1.output_scale)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def right_corner(self, state_at_t0, state_at_t1) -> _interp.InterpRes:
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(state_at_t1.strategy)

        prev = _common.State(prev_p, output_scale=state_at_t0.output_scale)
        sol = _common.State(sol_p, output_scale=state_at_t1.output_scale)
        acc = _common.State(acc_p, output_scale=state_at_t1.output_scale)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)


def _solver_flatten(solver):
    children = (solver.strategy,)
    aux = (solver.requires_rescaling, solver.string_repr)
    return children, aux


def _solver_unflatten(aux, children):
    (strategy,) = children
    rescaling, string_repr = aux
    return _UncalibratedSolver(
        strategy=strategy, requires_rescaling=rescaling, string_repr=string_repr
    )


tree_util.register_pytree_node(
    _UncalibratedSolver, flatten_func=_solver_flatten, unflatten_func=_solver_unflatten
)
