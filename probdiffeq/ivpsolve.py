"""Routines for estimating solutions of initial value problems."""

import functools
import warnings

import jax
import jax.numpy as jnp

from probdiffeq import _ivpsolve_impl
from probdiffeq.backend import tree_array_util
from probdiffeq.impl import impl
from probdiffeq.solvers import markov

# todo: change the Solution object to a simple
#  named tuple containing (t, full_estimate, u_and_marginals, stats).
#  No need to pre/append the initial condition to the solution anymore,
#  since the user knows it already.


class Solution:
    """Estimated initial value problem solution."""

    def __init__(
        self,
        t,
        u,
        output_scale,
        marginals,
        posterior,
        num_steps,
    ):
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
        if jnp.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access a single item of the solution."""
        if jnp.ndim(self.t) < 1:
            msg = "Solution object not batched :("
            raise ValueError(msg)

        if jnp.ndim(self.t) == 1 and item != -1:
            msg = "Access to non-terminal states is not available."
            raise ValueError(msg)

        return jax.tree_util.tree_map(lambda s: s[item, ...], self)

    def __iter__(self):
        """Iterate through the solution."""
        if jnp.ndim(self.t) <= 1:
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


jax.tree_util.register_pytree_node(Solution, _sol_flatten, _sol_unflatten)


def simulate_terminal_values(
    vector_field, initial_condition, t0, t1, adaptive_solver, dt0
) -> Solution:
    """Simulate the terminal values of an initial value problem."""
    save_at = jnp.asarray([t1])
    (_t, solution_save_at), _, num_steps = _ivpsolve_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        t0,
        initial_condition,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    # "squeeze"-type functionality (there is only a single state!)
    squeeze_fun = functools.partial(jnp.squeeze, axis=0)
    solution_save_at = jax.tree_util.tree_map(squeeze_fun, solution_save_at)
    num_steps = jax.tree_util.tree_map(squeeze_fun, num_steps)

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

    (_t, solution_save_at), _, num_steps = _ivpsolve_impl.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
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

    (t, solution_every_step), _dt, num_steps = _ivpsolve_impl.solve_and_save_every_step(
        jax.tree_util.Partial(vector_field),
        t0,
        initial_condition,
        t1=t1,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
    )
    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = jnp.concatenate((jnp.atleast_1d(t0), t))

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


def solve_fixed_grid(vector_field, initial_condition, grid, solver) -> Solution:
    """Solve an initial value problem on a fixed, pre-determined grid."""
    # Compute the solution
    _t, (posterior, output_scale) = _ivpsolve_impl.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        initial_condition,
        grid=grid,
        solver=solver,
    )

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
        num_steps=jnp.arange(1.0, len(grid)),
    )


def _userfriendly_output(*, posterior, posterior_t0):
    if isinstance(posterior, markov.MarkovSeq):
        # Compute marginals
        posterior_no_filter_marginals = markov.select_terminal(posterior)
        marginals = markov.marginals(posterior_no_filter_marginals, reverse=True)

        # Prepend the marginal at t1 to the computed marginals
        marginal_t1 = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # Prepend the marginal at t1 to the inits
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = markov.MarkovSeq(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior
