"""Routines for estimating solutions of initial value problems."""

from probdiffeq.backend import (
    control_flow,
    functools,
    tree_array_util,
    tree_util,
    warnings,
)
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.solvers import markov

# todo: change the Solution object to a simple
#  named tuple containing (t, full_estimate, u_and_marginals, stats).
#  No need to pre/append the initial condition to the solution anymore,
#  since the user knows it already.


class Solution:
    """Estimated initial value problem solution."""

    def __init__(self, t, u, output_scale, marginals, posterior, num_steps):
        """Construct a solution object."""
        self.t = t
        self.u = u
        self.output_scale = output_scale
        self.marginals = marginals
        self.posterior = posterior
        self.num_steps = num_steps

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


def _sol_flatten(sol):
    children = (
        sol.t,
        sol.u,
        sol.marginals,
        sol.posterior,
        sol.output_scale,
        sol.num_steps,
    )
    aux = ()
    return children, aux


def _sol_unflatten(_aux, children):
    t, u, marginals, posterior, output_scale, n = children
    return Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=n,
    )


tree_util.register_pytree_node(Solution, _sol_flatten, _sol_unflatten)


def simulate_terminal_values(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0
) -> Solution:
    """Simulate the terminal values of an initial value problem."""
    save_at = np.asarray([t1])
    (_t, solution_save_at), _, num_steps = _solve_and_save_at(
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
    posterior, output_scale = solution_save_at
    marginals = posterior.init if isinstance(posterior, markov.MarkovSeq) else posterior
    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=t1,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def solve_and_save_at(
    vector_field, initial_condition, save_at, adaptive_solver, dt0
) -> Solution:
    """Solve an initial value problem and return the solution at a pre-determined grid.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    if not adaptive_solver.solver.strategy.is_suitable_for_save_at:
        msg = (
            f"Strategy {adaptive_solver.solver.strategy} should not "
            f"be used in solve_and_save_at. "
        )
        warnings.warn(msg, stacklevel=1)

    (_t, solution_save_at), _, num_steps = _solve_and_save_at(
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
    _tmp = _userfriendly_output(posterior=posterior_save_at, posterior_t0=posterior_t0)
    marginals, posterior = _tmp
    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=save_at,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def _solve_and_save_at(
    vector_field, t, initial_condition, *, save_at, adaptive_solver, dt0
):
    advance_func = functools.partial(
        _advance_and_interpolate,
        vector_field=vector_field,
        adaptive_solver=adaptive_solver,
    )

    state = adaptive_solver.init(t, initial_condition, dt0=dt0, num_steps=0.0)
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
        return s.t + 10 * np.finfo_eps(float) < t_next

    def body_fun(s):
        return adaptive_solver.rejection_loop(s, vector_field=vector_field, t1=t_next)

    state = control_flow.while_loop(cond_fun, body_fun, init=state)

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    state, solution = control_flow.cond(
        state.t > t_next + 10 * np.finfo_eps(float),
        adaptive_solver.interpolate_and_extract,
        lambda s, _t: adaptive_solver.right_corner_and_extract(s),
        state,
        t_next,
    )
    return state, solution


def solve_and_save_every_step(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0
) -> Solution:
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    if not adaptive_solver.solver.strategy.is_suitable_for_save_every_step:
        msg = (
            f"Strategy {adaptive_solver.solver.strategy} should not "
            f"be used in solve_and_save_every_step."
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
    (t, solution_every_step), _dt, num_steps = tree_array_util.tree_stack(
        list(generator)
    )

    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = np.concatenate((np.atleast_1d(t0), t))

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    posterior, output_scale = solution_every_step
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0)
    marginals, posterior = _tmp

    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=t,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=num_steps,
    )


def _solution_generator(
    vector_field, t, initial_condition, *, dt0, t1, adaptive_solver
):
    """Generate a probabilistic IVP solution iteratively."""
    state = adaptive_solver.init(t, initial_condition, dt0=dt0, num_steps=0)

    while state.t < t1:
        state = adaptive_solver.rejection_loop(state, vector_field=vector_field, t1=t1)

        if state.t < t1:
            solution = adaptive_solver.extract(state)
            yield solution

    # Either interpolate (t > t_next) or "finalise" (t == t_next)
    if state.t > t1:
        _, solution = adaptive_solver.interpolate_and_extract(state, t=t1)
    else:
        _, solution = adaptive_solver.right_corner_and_extract(state)

    yield solution


def solve_fixed_grid(vector_field, initial_condition, grid, solver) -> Solution:
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
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0)
    marginals, posterior = _tmp

    u = impl.hidden_model.qoi(marginals)
    return Solution(
        t=grid,
        u=u,
        marginals=marginals,
        posterior=posterior,
        output_scale=output_scale,
        num_steps=np.arange(1.0, len(grid)),
    )


def _userfriendly_output(*, posterior, posterior_t0):
    if isinstance(posterior, markov.MarkovSeq):
        # Compute marginals
        posterior_no_filter_marginals = markov.select_terminal(posterior)
        marginals = markov.marginals(posterior_no_filter_marginals, reverse=True)

        # Prepend the marginal at t1 to the computed marginals
        marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # Prepend the marginal at t1 to the inits
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = markov.MarkovSeq(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior
