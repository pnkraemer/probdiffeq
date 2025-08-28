"""Routines for estimating solutions of initial value problems."""

from probdiffeq import stats
from probdiffeq.backend import (
    containers,
    control_flow,
    functools,
    linalg,
    tree_array_util,
    tree_util,
    warnings,
)
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array


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
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0, *, ssm
) -> IVPSolution:
    """Simulate the terminal values of an initial value problem."""
    save_at = np.asarray([t1])
    (_t, solution_save_at), _, num_steps = _solve_adaptive_save_at(
        tree_util.Partial(vector_field),
        t0,
        initial_condition,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    # "squeeze"-type functionality (there is only a single state!)
    squeeze_fun = functools.partial(np.squeeze_along_axis, axis=0)
    solution_save_at = tree_util.tree_map(squeeze_fun, solution_save_at)
    num_steps = tree_util.tree_map(squeeze_fun, num_steps)

    # I think the user expects marginals, so we compute them here
    # todo: do this in IVPSolution.* methods?
    posterior, output_scale = solution_save_at
    marginals = posterior.init if isinstance(posterior, stats.MarkovSeq) else posterior

    u = ssm.stats.qoi_from_sample(marginals.mean)
    std = ssm.stats.standard_deviation(marginals)
    u_std = ssm.stats.qoi_from_sample(std)
    return IVPSolution(
        t=t1,
        u=u,
        u_std=u_std,
        ssm=ssm,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_adaptive_save_at(
    vector_field, initial_condition, save_at, adaptive_solver, dt0, *, ssm
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
            author    = {Kr\"{a}mer, Nicholas},
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
    if not adaptive_solver.solver.is_suitable_for_save_at:
        msg = (
            f"Strategy {adaptive_solver.solver} should not "
            f"be used in solve_adaptive_save_at. "
        )
        warnings.warn(msg, stacklevel=1)

    (_t, solution_save_at), _, num_steps = _solve_adaptive_save_at(
        tree_util.Partial(vector_field),
        save_at[0],
        initial_condition,
        save_at=save_at[1:],
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )

    # I think the user expects the initial condition to be part of the state
    # (as well as marginals), so we compute those things here
    posterior_t0, *_ = initial_condition
    posterior_save_at, output_scale = solution_save_at
    _tmp = _userfriendly_output(
        posterior=posterior_save_at, posterior_t0=posterior_t0, ssm=ssm
    )
    marginals, posterior = _tmp
    u = ssm.stats.qoi_from_sample(marginals.mean)
    std = ssm.stats.standard_deviation(marginals)
    u_std = ssm.stats.qoi_from_sample(std)
    return IVPSolution(
        t=save_at,
        u=u,
        u_std=u_std,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
        ssm=ssm,
    )


def _solve_adaptive_save_at(
    vector_field, t, initial_condition, *, save_at, adaptive_solver, dt0
):
    advance_func = functools.partial(
        _advance_and_interpolate,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    state = adaptive_solver.init(t, initial_condition, dt=dt0, num_steps=0.0)
    _, solution = control_flow.scan(advance_func, init=state, xs=save_at, reverse=False)
    return solution


def _advance_and_interpolate(state, t_next, *, vector_field, adaptive_solver):
    # Advance until accepted.t >= t_next.
    # Note: This could already be the case and we may not loop (just interpolate)
    def cond_fun(s):
        # Terminate the loop if
        # the difference from s.t to t_next is smaller than a constant factor
        # (which is a "small" multiple of the current machine precision)
        # or if s.t > t_next holds.
        return s.step_from.t + 10 * np.finfo_eps(float) < t_next

    def body_fun(s):
        return adaptive_solver.rejection_loop(s, vector_field=vector_field, t1=t_next)

    state = control_flow.while_loop(cond_fun, body_fun, init=state)

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    state, solution = control_flow.cond(
        state.step_from.t > t_next + 10 * np.finfo_eps(float),
        adaptive_solver.extract_after_t1_via_interpolation,
        lambda s, _t: adaptive_solver.extract_at_t1(s),
        state,
        t_next,
    )
    return state, solution


def solve_adaptive_save_every_step(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0, *, ssm
) -> IVPSolution:
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    if not adaptive_solver.solver.is_suitable_for_save_every_step:
        msg = (
            f"Strategy {adaptive_solver.solver} should not "
            f"be used in solve_adaptive_save_every_step."
        )
        warnings.warn(msg, stacklevel=1)

    generator = _solution_generator(
        tree_util.Partial(vector_field),
        t0,
        initial_condition,
        t1=t1,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    tmp = tree_array_util.tree_stack(list(generator))
    (t, solution_every_step), _dt, num_steps = tmp

    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = np.concatenate((np.atleast_1d(t0), t))

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    posterior, output_scale = solution_every_step
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0, ssm=ssm)
    marginals, posterior = _tmp

    u = ssm.stats.qoi_from_sample(marginals.mean)
    std = ssm.stats.standard_deviation(marginals)
    u_std = ssm.stats.qoi_from_sample(std)
    return IVPSolution(
        t=t,
        u=u,
        u_std=u_std,
        ssm=ssm,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def _solution_generator(
    vector_field, t, initial_condition, *, dt0, t1, adaptive_solver
):
    state = adaptive_solver.init(t, initial_condition, dt=dt0, num_steps=0)

    while state.step_from.t < t1:
        state = adaptive_solver.rejection_loop(state, vector_field=vector_field, t1=t1)

        if state.step_from.t < t1:
            solution = adaptive_solver.extract_before_t1(state)
            yield solution

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if state.step_from.t > t1:
        _, solution = adaptive_solver.extract_after_t1_via_interpolation(state, t=t1)
    else:
        _, solution = adaptive_solver.extract_at_t1(state)

    yield solution


def solve_fixed_grid(
    vector_field, initial_condition, grid, solver, *, ssm
) -> IVPSolution:
    """Solve an initial value problem on a fixed, pre-determined grid."""
    # Compute the solution

    def body_fn(s, dt):
        _error, s_new = solver.step(state=s, vector_field=vector_field, dt=dt)
        return s_new, s_new

    t0 = grid[0]
    state0 = solver.init(t0, initial_condition)
    _, result_state = control_flow.scan(body_fn, init=state0, xs=np.diff(grid))
    _t, (posterior, output_scale) = solver.extract(result_state)

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0, ssm=ssm)
    marginals, posterior = _tmp

    u = ssm.stats.qoi_from_sample(marginals.mean)
    std = ssm.stats.standard_deviation(marginals)
    u_std = ssm.stats.qoi_from_sample(std)
    return IVPSolution(
        t=grid,
        u=u,
        u_std=u_std,
        ssm=ssm,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=np.arange(1.0, len(grid)),
    )


def _userfriendly_output(*, posterior, posterior_t0, ssm):
    if isinstance(posterior, stats.MarkovSeq):
        # Compute marginals
        posterior_no_filter_marginals = stats.markov_select_terminal(posterior)
        marginals = stats.markov_marginals(
            posterior_no_filter_marginals, reverse=True, ssm=ssm
        )

        # Prepend the marginal at t1 to the computed marginals
        marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # Prepend the marginal at t1 to the inits
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = stats.MarkovSeq(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior


def dt0(vf_autonomous, initial_values, /, scale=0.01, nugget=1e-5):
    """Propose an initial time-step."""
    u0, *_ = initial_values
    f0 = vf_autonomous(*initial_values)

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
    scale = atol + np.abs(y0) * rtol
    d0, d1 = linalg.vector_norm(y0), linalg.vector_norm(f0)

    dt0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + dt0 * f0
    f1 = vf(y1, t=t0 + dt0)
    d2 = linalg.vector_norm((f1 - f0) / scale) / dt0

    dt1 = np.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        np.maximum(1e-6, dt0 * 1e-3),
        (0.01 / np.maximum(d1, d2)) ** (1.0 / (error_contraction_rate + 1.0)),
    )
    return np.minimum(100.0 * dt0, dt1)
