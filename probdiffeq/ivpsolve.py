"""Routines for estimating solutions of initial value problems."""

from probdiffeq.backend import (
    containers,
    control_flow,
    linalg,
    tree_array_util,
    tree_util,
    warnings,
)
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array, TypeVar

T = TypeVar("T")


@containers.dataclass
class IVPSolution:
    """The probabilistic numerical solution of an initial value problem (IVP).

    This class stores the computed solution,
    its uncertainty estimates, and details of the probabilistic model
    used in probabilistic numerical integration.
    """

    t: Array
    """Time points at which the IVP solution has been computed."""

    u: Array
    """The mean of the IVP solution at each computed time point."""

    u_std: Array
    """The standard deviation of the IVP solution, indicating uncertainty."""

    output_scale: Array
    """The calibrated output scale of the probabilistic model."""

    marginals: Any
    """Marginal distributions for each time point in the posterior distribution."""

    posterior: Any
    """A the full posterior distribution of the probabilistic numerical solution.

    Typically, a backward factorisation of the posterior.
    """

    num_steps: Array
    """The number of solver steps taken at each time point."""

    ssm: Any
    """State-space model implementation used by the solver."""

    @staticmethod
    def _register_pytree_node():
        def _sol_flatten(sol):
            children = (
                sol.t,
                sol.u,
                sol.u_std,
                sol.marginals,
                sol.posterior,
                sol.output_scale,
                sol.num_steps,
            )
            aux = (sol.ssm,)
            return children, aux

        def _sol_unflatten(aux, children):
            (ssm,) = aux
            t, u, u_std, marginals, posterior, output_scale, n = children
            return IVPSolution(
                t=t,
                u=u,
                u_std=u_std,
                marginals=marginals,
                posterior=posterior,
                output_scale=output_scale,
                num_steps=n,
                ssm=ssm,
            )

        tree_util.register_pytree_node(IVPSolution, _sol_flatten, _sol_unflatten)


IVPSolution._register_pytree_node()


def solve_adaptive_terminal_values(
    ssm_init, /, *, t0, t1, adaptive_solver, dt0, ssm
) -> IVPSolution:
    """Simulate the terminal values of an initial value problem."""
    save_at = np.asarray([t0, t1])
    solution = solve_adaptive_save_at(
        ssm_init,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        ssm=ssm,
        warn=False,  # Turn off warnings because any solver goes for terminal values
    )
    return tree_util.tree_map(lambda s: s[-1], solution)


def solve_adaptive_save_at(
    ssm_init, /, *, save_at, adaptive, solver, dt0, ssm, warn=True
) -> IVPSolution:
    r"""Solve an initial value problem and return the solution at a pre-determined grid.

    This algorithm implements the method by Krämer (2024). Please consider citing it
    if you use it for your research. A PDF is available
    [here](https://arxiv.org/abs/2410.10530) and Krämer's (2024) experiments are
    available [here](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).

    ??? note "BibTex for Krämer (2024)"
        ```bibtex
        @InProceedings{kramer2024adaptive,
            title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory
                        Requirements},
            author    = {Kr{\"a}mer, Nicholas},
            booktitle = {Proceedings of the First International Conference on
                        Probabilistic Numerics},
            pages     = {12--24},
            year      = {2025},
            editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra
                        and Hennig, Philipp},
            volume    = {271},
            series    = {Proceedings of Machine Learning Research},
            publisher = {PMLR},
            url       = {https://proceedings.mlr.press/v271/kramer25a.html}
        }
        ```
    """
    if not solver.is_suitable_for_save_at and warn:
        msg = f"Strategy {solver} should not be used in solve_adaptive_save_at. "
        warnings.warn(msg, stacklevel=1)

    def advance(sol_and_state: T, t_next) -> tuple[T, Any]:
        """Advance the adaptive solver to the next checkpoint."""

        # Advance until accepted.t >= t_next.
        # Note: This could already be the case and we may not loop (just interpolate)
        @tree_util.register_dataclass
        @containers.dataclass
        class AdvanceState:
            do_continue: bool
            solution: Any
            adaptive: Any

        def cond_fun(c: AdvanceState) -> bool:
            return c.do_continue

        def body_fun(state: AdvanceState) -> AdvanceState:
            solution, state_new = adaptive.rejection_loop(
                state.adaptive, t1=t_next, solver=solver
            )
            do_continue = state_new.step_from.t + adaptive.eps < t_next
            return AdvanceState(do_continue, solution, state_new)

        init = AdvanceState(
            True, *sol_and_state
        )  # always step >=1x into the rejection loop
        advanced = control_flow.while_loop(cond_fun, body_fun, init=init)
        return (advanced.solution, advanced.adaptive), advanced.solution

    # Initialise the adaptive solver
    solution0 = solver.init(save_at[0], ssm_init)
    state = adaptive.init(solution0, dt=dt0)

    # Advance to one checkpoint after the other
    init = (solution0, state)
    xs = save_at[1:]
    (_solution, _state), solution = control_flow.scan(
        advance, init=init, xs=xs, reverse=False
    )

    # Stack the initial value into the solution and return
    result = solver.userfriendly_output(solution0=solution0, solution=solution)
    return IVPSolution(
        t=result.t,
        u=result.estimate.u,
        u_std=result.estimate.u_std,
        marginals=result.estimate.marginals,
        posterior=result.posterior,
        output_scale=result.output_scale,
        num_steps=result.num_steps,
        ssm=ssm,
    )


def solve_adaptive_save_every_step(
    ssm_init, /, *, t0, t1, adaptive, solver, dt0, ssm
) -> IVPSolution:
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    if not solver.is_suitable_for_save_every_step:
        msg = f"Strategy {solver} should not be used in solve_adaptive_save_every_step."
        warnings.warn(msg, stacklevel=1)

    t0, t1 = np.asarray(t0), np.asarray(t1)
    solution0 = solver.init(t0, ssm_init)
    state = adaptive.init(solution0, dt=dt0)
    solutions = []
    while state.step_from.t < t1:
        solution, state = adaptive.rejection_loop(state, t1=t1, solver=solver)

        solutions.append(solution)

    #     if state.step_from.t + adaptive.eps < t1:
    #         _, solution = adaptive.extract_before_t1(state, t=t1)
    # # Either interpolate (t > t_next) or "finalise" (t == t_next)
    # is_after_t1 = state.step_from.t > t1 + adaptive.eps
    # if is_after_t1:
    #     _, solution = adaptive.extract_after_t1(state, t=t1)
    # else:
    #     _, solution = adaptive.extract_at_t1(state, t=t1)
    # solutions.append(solution)

    solutions = tree_array_util.tree_stack(solutions)
    solutions = solver.userfriendly_output(solution0=solution0, solution=solutions)
    return IVPSolution(
        t=solutions.t,
        u=solutions.estimate.u,
        u_std=solutions.estimate.u_std,
        marginals=solutions.estimate.marginals,
        posterior=solutions.posterior,
        output_scale=solutions.output_scale,
        num_steps=solutions.num_steps,
        ssm=ssm,
    )


def solve_fixed_grid(ssm_init, /, *, grid, solver, ssm) -> IVPSolution:
    """Solve an initial value problem on a fixed, pre-determined grid."""
    # Compute the solution

    def body_fn(s, dt):
        _error, s_new = solver.step(state=s, dt=dt)
        return s_new, s_new

    t0 = grid[0]
    state0 = solver.init(t0, ssm_init)
    _, result = control_flow.scan(body_fn, init=state0, xs=np.diff(grid))

    result = solver.userfriendly_output(solution0=state0, solution=result)

    return IVPSolution(
        t=result.t,
        u=result.estimate.u,
        u_std=result.estimate.u_std,
        marginals=result.estimate.marginals,
        posterior=result.posterior,
        output_scale=result.output_scale,
        num_steps=result.num_steps,
        ssm=ssm,
    )


def dt0(vf_autonomous, initial_values, /, scale=0.01, nugget=1e-5):
    """Propose an initial time-step."""
    u0, *_ = initial_values
    f0 = vf_autonomous(*initial_values)

    u0, _ = tree_util.ravel_pytree(u0)
    f0, _ = tree_util.ravel_pytree(f0)

    norm_y0 = linalg.vector_norm(u0)
    norm_dy0 = linalg.vector_norm(f0) + nugget

    return scale * norm_y0 / norm_dy0


def dt0_adaptive(vf, initial_values, /, t0, *, error_contraction_rate, rtol, atol):
    """Propose an initial time-step as a function of the tolerances."""
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    # Implementation mostly copied from
    #
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #

    if len(initial_values) > 1:
        raise ValueError
    y0 = initial_values[0]

    f0 = vf(y0, t=t0)

    y0, unravel = tree_util.ravel_pytree(y0)
    f0, _ = tree_util.ravel_pytree(f0)

    scale = atol + np.abs(y0) * rtol
    d0, d1 = linalg.vector_norm(y0), linalg.vector_norm(f0)

    dt0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + dt0 * f0
    f1 = vf(unravel(y1), t=t0 + dt0)
    f1, _ = tree_util.ravel_pytree(f1)
    d2 = linalg.vector_norm((f1 - f0) / scale) / dt0

    dt1 = np.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        np.maximum(1e-6, dt0 * 1e-3),
        (0.01 / np.maximum(d1, d2)) ** (1.0 / (error_contraction_rate + 1.0)),
    )
    return np.minimum(100.0 * dt0, dt1)
