"""Uncalibrated IVP solvers."""
import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.ivpsolvers import _common
from probdiffeq.ivpsolvers import solver as solver_module


def solver(strategy, _unneeded):
    """Create a solver that does not calibrate the output scale automatically."""
    string_repr = f"<Uncalibrated solver with {strategy}>"
    return UncalibratedSolver(
        strategy=strategy, string_repr=string_repr, requires_rescaling=False
    )


class UncalibratedSolver(solver_module.Solver[_common.State]):
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
        state_strategy = self.strategy.begin(
            state.strategy,
            dt=dt,
            parameters=parameters,
            vector_field=vector_field,
        )
        (error, _, cache_obs) = state_strategy.corr
        state_strategy = self.strategy.complete(
            state_strategy,
            parameters=parameters,
            vector_field=vector_field,
            output_scale=state.output_scale,
        )
        # Extract and return solution
        return _common.State(
            error_estimate=dt * error,
            strategy=state_strategy,
            output_scale=state.output_scale,
            num_steps=state.num_steps + 1,
        )

    def extract(self, state: _common.State, /):
        t, posterior = self.strategy.extract(state.strategy)
        return t, posterior, state.output_scale, state.num_steps

    def interpolate(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            output_scale=s1.output_scale,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def right_corner(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            output_scale=s1.output_scale,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return _interp.InterpRes(accepted=acc, solution=sol, previous=prev)

    def _interp_make_state(
        self, state_strategy, *, reference: _common.State
    ) -> _common.State:
        error_estimate = jnp.empty_like(state_strategy.u)
        return _common.State(
            strategy=state_strategy,
            error_estimate=error_estimate,
            output_scale=reference.output_scale,
            num_steps=reference.num_steps,
        )


def _solver_flatten(solver):
    children = (solver.strategy,)
    aux = (solver.requires_rescaling, solver.string_repr)
    return children, aux


def _solver_unflatten(aux, children):
    (strategy,) = children
    rescaling, string_repr = aux
    return UncalibratedSolver(
        strategy=strategy, requires_rescaling=rescaling, string_repr=string_repr
    )


jax.tree_util.register_pytree_node(
    nodetype=UncalibratedSolver,
    flatten_func=_solver_flatten,
    unflatten_func=_solver_unflatten,
)
