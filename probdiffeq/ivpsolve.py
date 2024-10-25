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
from probdiffeq.backend.typing import Any, Callable, NamedArg


@containers.dataclass
class _Controller:
    """Control algorithm."""

    init: Callable[[float], Any]
    """Initialise the controller state."""

    clip: Callable[[Any, float, float], Any]
    """(Optionally) clip the current step to not exceed t1."""

    apply: Callable[[Any, NamedArg(float, "error_power")], Any]
    r"""Propose a time-step $\Delta t$."""

    extract: Callable[[Any], float]
    """Extract the time-step from the controller state."""


def control_proportional_integral(
    *,
    clip: bool = False,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _Controller:
    """Construct a proportional-integral-controller with time-clipping."""

    class PIState(containers.NamedTuple):
        dt: float
        error_power_previously_accepted: float

    def init(dt: float, /) -> PIState:
        return PIState(dt, 1.0)

    def apply(state: PIState, /, *, error_power) -> PIState:
        dt_proposed, error_power_prev = state
        # error_power = error_norm ** (-1.0 / error_contraction_rate)

        a1 = error_power**power_integral_unscaled
        a2 = (error_power / error_power_prev) ** power_proportional_unscaled
        scale_factor_unclipped = safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt_proposed
        return PIState(dt_proposed, error_power_prev)

    def extract(state: PIState, /) -> float:
        dt_proposed, _error_norm_previously_accepted = state
        return dt_proposed

    if clip:

        def clip_fun(state: PIState, /, t, t1) -> PIState:
            dt_proposed, error_norm_previously_accepted = state
            dt = dt_proposed
            dt_clipped = np.minimum(dt, t1 - t)
            return PIState(dt_clipped, error_norm_previously_accepted)

        return _Controller(init=init, apply=apply, extract=extract, clip=clip_fun)

    return _Controller(init=init, apply=apply, extract=extract, clip=lambda v, **_kw: v)


def control_integral(
    *, clip=False, safety=0.95, factor_min=0.2, factor_max=10.0
) -> _Controller:
    """Construct an integral-controller."""

    def init(dt, /):
        return dt

    def apply(dt, /, *, error_power):
        # error_power = error_norm ** (-1.0 / error_contraction_rate)
        scale_factor_unclipped = safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
        return scale_factor * dt

    def extract(dt, /):
        return dt

    if clip:

        def clip_fun(dt, /, t, t1):
            return np.minimum(dt, t1 - t)

        return _Controller(init=init, apply=apply, extract=extract, clip=clip_fun)

    return _Controller(init=init, apply=apply, extract=extract, clip=lambda v, **_kw: v)


def adaptive(solver, *, ssm, atol=1e-4, rtol=1e-2, control=None, norm_ord=None):
    """Make an IVP solver adaptive."""
    if control is None:
        control = control_proportional_integral()

    return _AdaSolver(
        solver, ssm=ssm, atol=atol, rtol=rtol, control=control, norm_ord=norm_ord
    )


class _AdaState(containers.NamedTuple):
    step_from: Any
    interp_from: Any
    control: Any
    stats: Any


class _AdaSolver:
    """Adaptive IVP solvers."""

    def __init__(self, solver, *, atol, rtol, control, norm_ord, ssm):
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.ssm = ssm

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

    @functools.jit
    def init(self, t, initial_condition, dt, num_steps) -> _AdaState:
        """Initialise the IVP solver state."""
        state_solver = self.solver.init(t, initial_condition)
        state_control = self.control.init(dt)
        return _AdaState(state_solver, state_solver, state_control, num_steps)

    @functools.jit
    def rejection_loop(self, state0: _AdaState, *, vector_field, t1) -> _AdaState:
        class _RejectionState(containers.NamedTuple):
            """State for rejection loops.

            (Keep decreasing step-size until error norm is small.
            This is one part of an IVP solver step.)
            """

            error_norm_proposed: float
            control: Any
            proposed: Any
            step_from: Any

        def init(s0: _AdaState) -> _RejectionState:
            def _inf_like(tree):
                return tree_util.tree_map(lambda x: np.inf() * np.ones_like(x), tree)

            larger_than_1 = 1.1
            return _RejectionState(
                error_norm_proposed=larger_than_1,
                control=s0.control,
                proposed=_inf_like(s0.step_from),
                step_from=s0.step_from,
            )

        def cond_fn(state: _RejectionState) -> bool:
            return state.error_norm_proposed > 1.0

        def body_fn(state: _RejectionState) -> _RejectionState:
            """Attempt a step.

            Perform a step with an IVP solver and
            propose a future time-step based on tolerances and error estimates.
            """
            # Some controllers like to clip the terminal value instead of interpolating.
            # This must happen _before_ the step.
            state_control = self.control.clip(state.control, t=state.step_from.t, t1=t1)

            # Perform the actual step.
            # todo: error estimate should be a tuple (abs, rel)
            error_estimate, state_proposed = self.solver.step(
                state=state.step_from,
                vector_field=vector_field,
                dt=self.control.extract(state_control),
            )
            # Normalise the error
            u_proposed = self.ssm.stats.qoi(state_proposed.strategy.hidden)[0]
            u_step_from = self.ssm.stats.qoi(state_proposed.strategy.hidden)[0]
            u = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
            error_norm = _normalise_error(error_estimate, u=u)

            # Propose a new step
            error_power = error_norm ** (-1.0 / self.solver.error_contraction_rate)
            state_control = self.control.apply(state_control, error_power=error_power)
            return _RejectionState(
                error_norm_proposed=error_norm,  # new
                proposed=state_proposed,  # new
                control=state_control,  # new
                step_from=state.step_from,
            )

        def _normalise_error(error_estimate, *, u):
            error_relative = error_estimate / (self.atol + self.rtol * np.abs(u))
            dim = np.atleast_1d(u).size
            error_norm = linalg.vector_norm(error_relative, order=self.norm_ord)
            return error_norm / np.sqrt(dim)

        def extract(s: _RejectionState) -> _AdaState:
            num_steps = state0.stats + 1
            return _AdaState(s.proposed, s.step_from, s.control, num_steps)

        init_val = init(state0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def extract_before_t1(self, state: _AdaState):
        solution_solver = self.solver.extract(state.step_from)
        solution_control = self.control.extract(state.control)
        return solution_solver, solution_control, state.stats

    def extract_at_t1(self, state: _AdaState):
        # todo: make the "at t1" decision inside interpolate(),
        #  which collapses the next two functions together
        interp = self.solver.interpolate_at_t1(
            interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            interp.step_from, interp.interp_from, state.control, state.stats
        )

        solution_solver = self.solver.extract(interp.interpolated)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)

    def extract_after_t1_via_interpolation(self, state: _AdaState, t):
        interp = self.solver.interpolate(
            t, interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            interp.step_from, interp.interp_from, state.control, state.stats
        )

        solution_solver = self.solver.extract(interp.interpolated)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)

    @staticmethod
    def register_pytree_node():
        def _asolver_flatten(asolver):
            children = (asolver.atol, asolver.rtol)
            aux = (asolver.solver, asolver.control, asolver.norm_ord, asolver.ssm)
            return children, aux

        def _asolver_unflatten(aux, children):
            atol, rtol = children
            (solver, control, norm_ord, ssm) = aux
            return _AdaSolver(
                solver=solver,
                atol=atol,
                rtol=rtol,
                control=control,
                norm_ord=norm_ord,
                ssm=ssm,
            )

        tree_util.register_pytree_node(
            _AdaSolver, flatten_func=_asolver_flatten, unflatten_func=_asolver_unflatten
        )


_AdaSolver.register_pytree_node()


class _Solution:
    """Estimated initial value problem solution."""

    def __init__(self, t, u, u_std, output_scale, marginals, posterior, num_steps, ssm):
        """Construct a solution object."""
        self.t = t
        self.u = u
        self.u_std = u_std
        self.output_scale = output_scale
        self.marginals = marginals  # todo: marginals are replaced by "u" and "u_std"
        self.posterior = posterior
        self.num_steps = num_steps
        self.ssm = ssm

    def __repr__(self):
        """Evaluate a string-representation of the solution object."""
        return (
            f"{self.__class__.__name__}("
            f"t={self.t},"
            f"u={self.u},"
            f"output_scale={self.output_scale},"
            f"marginals={self.marginals},"
            f"posterior={self.posterior},"
            f"num_steps={self.num_steps},"
            ")"
        )

    def __len__(self):
        """Evaluate the length of a solution."""
        if np.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access a single item of the solution."""
        if np.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)

        if np.ndim(self.t) == 1 and item != -1:
            msg = "Access to non-terminal states is not available."
            raise ValueError(msg)

        return tree_util.tree_map(lambda s: s[item, ...], self)

    def __iter__(self):
        """Iterate through the solution."""
        if np.ndim(self.t) <= 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)

        for i in range(self.t.shape[0]):
            yield self[i]

    @staticmethod
    def register_pytree_node():
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
            return _Solution(
                t=t,
                u=u,
                u_std=u_std,
                marginals=marginals,
                posterior=posterior,
                output_scale=output_scale,
                num_steps=n,
                ssm=ssm,
            )

        tree_util.register_pytree_node(_Solution, _sol_flatten, _sol_unflatten)


_Solution.register_pytree_node()


def solve_adaptive_terminal_values(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0, *, ssm
) -> _Solution:
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
    # todo: do this in _Solution.* methods?
    posterior, output_scale = solution_save_at
    marginals = posterior.init if isinstance(posterior, stats.MarkovSeq) else posterior

    u = ssm.stats.qoi_from_sample(marginals.mean)
    u_std = ssm.stats.qoi_from_sample(marginals.cholesky)
    return _Solution(
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
) -> _Solution:
    r"""Solve an initial value problem and return the solution at a pre-determined grid.

    This algorithm implements the method by Krämer (2024).
    Please consider citing it if you use it for your research.
    A PDF is available [here](https://arxiv.org/abs/2410.10530)
    and Krämer's (2024) experiments are
    [here](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).


    ??? note "BibTex for Krämer (2024)"
        ```bibtex
        @article{krämer2024adaptive,
            title={Adaptive Probabilistic {ODE} Solvers Without
            Adaptive Memory Requirements},
            author={Kr{\"a}mer, Nicholas},
            year={2024},
            eprint={2410.10530},
            archivePrefix={arXiv},
            url={https://arxiv.org/abs/2410.10530},
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
    u_std = ssm.stats.qoi_from_sample(marginals.cholesky)
    return _Solution(
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
) -> _Solution:
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
    u_std = ssm.stats.qoi_from_sample(marginals.cholesky)
    return _Solution(
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
    """Generate a probabilistic IVP solution iteratively."""
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
) -> _Solution:
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
    u_std = ssm.stats.qoi_from_sample(marginals.cholesky)
    return _Solution(
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
