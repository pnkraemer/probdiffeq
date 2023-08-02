"""IVP-solver API."""


import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.ivpsolvers import _common


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
        posterior = self.strategy.solution_from_tcoeffs(tcoeffs)
        return t, posterior, output_scale, num_steps

    def init(self, t, posterior, /, output_scale, num_steps) -> _common.State:
        state_strategy = self.strategy.init(t, posterior)
        error_estimate = jnp.empty_like(state_strategy.u)
        output_scale = self.calibration.init(output_scale)
        return _common.State(
            error_estimate=error_estimate,
            strategy=state_strategy,
            output_scale_prior=output_scale,
            output_scale_calibrated=output_scale,
            num_steps=num_steps,
        )

    def step(
        self, *, state: _common.State, vector_field, dt, parameters
    ) -> _common.State:
        return self._step_fun(
            state, dt, parameters, vector_field, strategy=self.strategy
        )

    def extract(self, state: _common.State, /):
        t, posterior = self.strategy.extract(state.strategy)
        output_scale = self.calibration.extract(state.output_scale_calibrated)
        return t, posterior, output_scale, state.num_steps

    def interpolate_fun(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
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

    def right_corner_fun(
        self, t, s0: _common.State, s1: _common.State
    ) -> _interp.InterpRes[_common.State]:
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

    def _interp_make_state(
        self, state_strategy, *, reference: _common.State
    ) -> _common.State:
        error_estimate = jnp.empty_like(state_strategy.u)
        return _common.State(
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
