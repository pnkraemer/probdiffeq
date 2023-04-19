"""Adaptive solvers for initial value problems (IVPs)."""
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import controls, ivpsolvers

S = TypeVar("S")
"""A type-variable for generic IVP solver states."""

C = TypeVar("C", bound=controls.Control)
"""A type-variable for generic controller states."""

T = TypeVar("T", bound=ivpsolvers.Solver)
"""A type-variable for (non-adaptive) IVP solvers."""


# basically a namedtuple, but NamedTuples cannot be generic,
#  which is why we implement this functionality manually.
@jax.tree_util.register_pytree_node_class
class _AdaptiveState(Generic[S, C]):
    """Adaptive IVP solver state."""

    def __init__(
        self,
        error_norm_proposed,
        control: C,
        proposed: S,
        accepted: S,
        solution: S,
    ):
        self.error_norm_proposed = error_norm_proposed
        self.control = control
        self.proposed = proposed
        self.accepted = accepted
        self.solution = solution

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"\n\terror_norm_proposed={self.error_norm_proposed},"
            f"\n\tcontrol={self.control},"
            f"\n\tproposed={self.proposed},"
            f"\n\taccepted={self.accepted},"
            f"\n\tsolution={self.solution},"
            "\n)"
        )

    def tree_flatten(self):
        children = (
            self.error_norm_proposed,
            self.control,
            self.proposed,
            self.accepted,
            self.solution,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (
            error_norm_proposed,
            control,
            proposed,
            accepted,
            solution,
        ) = children
        return cls(
            error_norm_proposed=error_norm_proposed,
            control=control,
            proposed=proposed,
            accepted=accepted,
            solution=solution,
        )


def _reference_state_fn_max_abs(sol, sol_previous):
    return jnp.maximum(jnp.abs(sol), jnp.abs(sol_previous))


@jax.tree_util.register_pytree_node_class
class AdaptiveIVPSolver(Generic[T]):
    """Adaptive IVP solvers."""

    def __init__(
        self,
        solver: T,
        atol=1e-4,
        rtol=1e-2,
        control=None,
        norm_ord=None,
        numerical_zero=1e-10,
        while_loop_fn=jax.lax.while_loop,
        reference_state_fn=_reference_state_fn_max_abs,
    ):
        if control is None:
            control = controls.ProportionalIntegral()

        self.solver = solver
        self.while_loop_fn = while_loop_fn
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.numerical_zero = numerical_zero
        self.reference_state_fn = reference_state_fn

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            f"\n\tnumerical_zero={self.numerical_zero},"
            f"\n\treference_state_fn={self.reference_state_fn},"
            "\n)"
        )

    def tree_flatten(self):
        children = (
            self.solver,
            self.atol,
            self.rtol,
            self.control,
            self.numerical_zero,
        )
        aux = self.norm_ord, self.reference_state_fn, self.while_loop_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        solver, atol, rtol, control, numerical_zero = children
        norm_ord, reference_state_fn, while_loop_fn = aux
        return cls(
            solver=solver,
            while_loop_fn=while_loop_fn,
            atol=atol,
            rtol=rtol,
            control=control,
            numerical_zero=numerical_zero,
            norm_ord=norm_ord,
            reference_state_fn=reference_state_fn,
        )

    @property
    def error_contraction_rate(self):
        """Error order."""
        # call self.solver.num_derivatives() ?
        return self.solver.strategy.extrapolation.num_derivatives + 1

    @jax.jit
    def init(self, sol, control_sol):
        """Initialise the IVP solver state."""
        (t, posterior, t_accepted, posterior_accepted, output_scale, num_steps) = sol
        # Initialise the components
        state_solver = self.solver.init(t, posterior, output_scale, num_steps)
        state_solver_accepted = self.solver.init(
            t_accepted, posterior_accepted, output_scale, num_steps
        )
        state_solver_proposed = _inf_like(state_solver)
        state_control = self.control.init_state_from_dt(*control_sol)

        # Initialise (prototypes for) proposed values
        error_norm_proposed = _inf_like(jnp.zeros((), dtype=float))
        print(f"Initialising at {t}")
        return _AdaptiveState(
            error_norm_proposed=error_norm_proposed,
            accepted=state_solver_accepted,
            proposed=state_solver_proposed,
            solution=state_solver,
            control=state_control,
        )

    @jax.jit
    def step(self, state, vector_field, t1, parameters):
        """Perform a full step (including acceptance/rejection)."""
        print("Entering step")
        enter_rejection_loop = state.accepted.t + self.numerical_zero < t1
        state = jax.lax.cond(
            enter_rejection_loop,
            lambda s: self._rejection_loop(
                state0=s,
                vector_field=vector_field,
                t1=t1,
                parameters=parameters,
            ),
            lambda s: s,
            state,
        )
        print(f"Am at {state.accepted.t, state.solution.t}")

        # todo: which output scale does this interpolation use?
        #  the MLE solver should use the prior one, but it looks like
        #  we are using the calibrated scale.
        #  to test this, assert that MLESolver and calibrated_solver(mle) are IDENTICAL.
        #  they will not be if the configuration is such that interpolation matters.
        state = jax.lax.cond(
            state.accepted.t + self.numerical_zero >= t1,
            lambda s: self._interpolate(state=s, t=t1),
            lambda s: self._accepted_as_solution(state=s),
            state,
        )
        return state

    def _rejection_loop(self, *, vector_field, state0, t1, parameters):
        print(f"Entering rejection loop from {state0.accepted.t} to maximally {t1}")

        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self._attempt_step(
                state=s,
                vector_field=vector_field,
                t1=t1,
                parameters=parameters,
            )
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init(s):
            return True, s

        _proceed, state_new = self.while_loop_fn(cond_fn, body_fn, init(state0))
        return _AdaptiveState(
            error_norm_proposed=_inf_like(state_new.error_norm_proposed),
            proposed=_inf_like(state_new.proposed),  # meaningless?
            accepted=state_new.proposed,  # holla! New! :)
            solution=state0.accepted,  # Not yet by interpolate() if necessary
            control=state_new.control,
        )

    def _attempt_step(self, *, state, vector_field, t1, parameters):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        state_control = self.control.clip(
            t=state.accepted.t, state=state.control, t1=t1
        )

        # Perform the actual step.
        posterior = self.solver.step(
            state=state.accepted,
            vector_field=vector_field,
            dt=state_control.dt_proposed,
            parameters=parameters,
        )
        # Normalise the error and propose a new step.
        error_normalised = self._normalise_error(
            error_estimate=posterior.error_estimate,
            u=self.reference_state_fn(posterior.u, state.accepted.u),
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        # print(f"Error norm previous {state_control.error_norm_previously_accepted} and now {error_normalised}")
        state_control = self.control.apply(
            state=state_control,
            error_normalised=error_normalised,
            error_contraction_rate=self.error_contraction_rate,
        )
        print("Proposed control:", state_control)
        return _AdaptiveState(
            error_norm_proposed=error_normalised,  # new
            proposed=posterior,  # new
            solution=state.solution,  # too early to accept :)
            accepted=state.accepted,  # too early to accept :)
            control=state_control,  # new
        )

    @staticmethod
    def _normalise_error(*, error_estimate, u, atol, rtol, norm_ord):
        error_relative = error_estimate / (atol + rtol * jnp.abs(u))
        dim = jnp.atleast_1d(u).size
        return jnp.linalg.norm(error_relative, ord=norm_ord) / jnp.sqrt(dim)

    def _interpolate(self, *, state: _AdaptiveState, t) -> _AdaptiveState:
        print(
            f"Interpolating from {state.solution.t} to {t} to {state.accepted.t} or stepped right to solution!"
        )
        accepted, solution = self.solver.interpolate(
            s0=state.solution, s1=state.accepted, t=t
        )
        return _AdaptiveState(
            error_norm_proposed=_inf_like(state.error_norm_proposed),
            proposed=_inf_like(state.proposed),
            accepted=accepted,  # holla! New! :)
            solution=solution,  # holla! New! :)
            control=state.control,
        )

    def _accepted_as_solution(self, state):
        return _AdaptiveState(
            error_norm_proposed=_inf_like(state.error_norm_proposed),
            proposed=_inf_like(state.proposed),
            accepted=state.accepted,
            solution=state.accepted,  # holla! New! :)
            control=state.control,
        )

    def extract(self, state: _AdaptiveState, /):
        print(f"Extracting at {state.solution.t}")
        print()
        (t, posterior, output_scale, num_steps) = self.solver.extract(state.solution)
        (t_accepted, posterior_accepted, *_) = self.solver.extract(state.accepted)
        control_extract = self.control.extract_dt_from_state(state.control)
        solution = (
            t,
            posterior,
            t_accepted,
            posterior_accepted,
            output_scale,
            num_steps,
        )
        return solution, control_extract


# todo: rename to "meaningless_like" or whatever
def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
