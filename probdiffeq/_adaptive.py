"""Adaptive solvers for initial value problems (IVPs)."""
from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq import _solver, controls
from probdiffeq.backend import containers, control_flow
from probdiffeq.impl import impl


class _RejectionState(containers.NamedTuple):
    """State for rejection loops.

    (Keep decreasing step-size until error norm is small.
    This is one part of an IVP solver step.)
    """

    error_norm_proposed: float
    control: Any
    proposed: Any
    step_from: Any


class AdaptiveIVPSolver:
    """Adaptive IVP solvers."""

    def __init__(
        self,
        solver: _solver.Solver,
        atol=1e-4,
        rtol=1e-2,
        control=None,
        norm_ord=None,
    ):
        if control is None:
            control = controls.proportional_integral()

        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            "\n)"
        )

    @jax.jit
    def init(self, t, posterior, output_scale, num_steps, dt0):
        """Initialise the IVP solver state."""
        state_solver = self.solver.init(t, posterior, output_scale, num_steps)
        state_control = self.control.init(dt0)
        return state_solver, state_control

    @jax.jit
    def rejection_loop(self, state0, control0, *, vector_field, t1, parameters):
        def cond_fn(s):
            return s.error_norm_proposed > 1.0

        def body_fn(s):
            return self._attempt_step(
                state=s,
                vector_field=vector_field,
                t1=t1,
                parameters=parameters,
            )

        def init(s0, c0):
            larger_than_1 = 1.1
            return _RejectionState(
                error_norm_proposed=larger_than_1,
                control=c0,
                proposed=_inf_like(s0),
                step_from=s0,
            )

        def extract(s):
            return s.proposed, s.control

        init_val = init(state0, control0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def _attempt_step(self, *, state: _RejectionState, vector_field, t1, parameters):
        """Attempt a step.

        Perform a step with an IVP solver and
        propose a future time-step based on tolerances and error estimates.
        """
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        state_control = self.control.clip(state.control, t=state.step_from.t, t1=t1)

        # Perform the actual step.
        state_proposed = self.solver.step(
            state=state.step_from,
            vector_field=vector_field,
            dt=self.control.extract(state_control),
            parameters=parameters,
        )
        # Normalise the error and propose a new step.
        u_proposed = impl.random.qoi(state_proposed.strategy.hidden)
        u_step_from = impl.random.qoi(state_proposed.strategy.hidden)
        u = jnp.maximum(jnp.abs(u_proposed), jnp.abs(u_step_from))
        error_normalised = self._normalise_error(
            error_estimate=state_proposed.error_estimate,
            u=u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        error_contraction_rate = self.solver.strategy.extrapolation.num_derivatives + 1
        state_control = self.control.apply(
            state_control,
            error_normalised=error_normalised,
            error_contraction_rate=error_contraction_rate,
        )
        return _RejectionState(
            error_norm_proposed=error_normalised,  # new
            proposed=state_proposed,  # new
            control=state_control,  # new
            step_from=state.step_from,
        )

    @staticmethod
    def _normalise_error(*, error_estimate, u, atol, rtol, norm_ord):
        error_relative = error_estimate / (atol + rtol * jnp.abs(u))
        dim = jnp.atleast_1d(u).size
        return jnp.linalg.norm(error_relative, ord=norm_ord) / jnp.sqrt(dim)

    def extract(self, accepted, control, /):
        solver_extract = self.solver.extract(accepted)
        control_extract = self.control.extract(control)
        return solver_extract, control_extract


# Register outside of class to declutter the AdaptiveIVPSolver source code a bit


def _asolver_flatten(asolver: AdaptiveIVPSolver):
    children = (asolver.solver, asolver.atol, asolver.rtol, asolver.control)
    aux = (asolver.norm_ord,)
    return children, aux


def _asolver_unflatten(aux, children):
    solver, atol, rtol, control = children
    (norm_ord,) = aux
    return AdaptiveIVPSolver(
        solver=solver,
        atol=atol,
        rtol=rtol,
        control=control,
        norm_ord=norm_ord,
    )


jax.tree_util.register_pytree_node(
    nodetype=AdaptiveIVPSolver,
    flatten_func=_asolver_flatten,
    unflatten_func=_asolver_unflatten,
)


def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
