"""Routines for estimating solutions of initial value problems."""

import functools
import warnings
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _adaptive, _collocate
from probdiffeq.backend import tree_array_util
from probdiffeq.impl import impl
from probdiffeq.solvers import markov

# TODO: make adaptive_solver and initial_condition arguments to the solver!


def simulate_terminal_values(
    vector_field,
    taylor_coefficients,
    t0,
    t1,
    solver,
    output_scale,
    dt0,
    **adaptive_solver_options,
):
    """Simulate the terminal values of an initial value problem."""
    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver, **adaptive_solver_options)
    initial_condition = solver.solution_from_tcoeffs(
        taylor_coefficients, output_scale=output_scale
    )

    save_at = jnp.asarray([t1])
    posterior, output_scale, num_steps = _collocate.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        t0,
        *initial_condition,
        save_at=save_at,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        interpolate=(solver.interpolate, solver.right_corner),
    )
    # "squeeze"-type functionality (there is only a single state!)
    squeeze_fun = functools.partial(jnp.squeeze, axis=0)
    posterior = jax.tree_util.tree_map(squeeze_fun, posterior)
    output_scale = jax.tree_util.tree_map(squeeze_fun, output_scale)
    num_steps = jax.tree_util.tree_map(squeeze_fun, num_steps)

    # I think the user expects marginals, so we compute them here
    if isinstance(posterior, markov.MarkovSeqRev):
        marginals = posterior.init
    else:
        marginals = posterior
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
    vector_field,
    taylor_coefficients,
    save_at,
    solver,
    output_scale,
    dt0,
    **adaptive_solver_options,
):
    """Solve an initial value problem and return the solution at a pre-determined grid.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    if not solver.strategy.is_suitable_for_save_at:
        msg = "Strategy {solver.strategy} cannot be used in save_at mode. "
        warnings.warn(msg, stacklevel=1)

    adaptive_solver = _adaptive.AdaptiveIVPSolver(solver, **adaptive_solver_options)

    initial_condition = solver.solution_from_tcoeffs(
        taylor_coefficients, output_scale=output_scale
    )

    posterior, output_scale, num_steps = _collocate.solve_and_save_at(
        jax.tree_util.Partial(vector_field),
        save_at[0],
        *initial_condition,
        save_at=save_at[1:],
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        interpolate=(solver.interpolate, solver.right_corner),
    )

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
    _tmp = _userfriendly_output(posterior=posterior, posterior_t0=posterior_t0)
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
    vector_field,
    taylor_coefficients,
    t0,
    t1,
    solver,
    output_scale,
    dt0,
    **adaptive_solver_options,
):
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.
    """
    adaptive_solver = _adaptive.AdaptiveIVPSolver(
        solver=solver, **adaptive_solver_options
    )
    initial_condition = solver.solution_from_tcoeffs(
        taylor_coefficients, output_scale=output_scale
    )

    t, posterior, output_scale, num_steps = _collocate.solve_and_save_every_step(
        jax.tree_util.Partial(vector_field),
        t0,
        *initial_condition,
        t1=t1,
        adaptive_solver=adaptive_solver,
        dt0=dt0,
        interpolate=(solver.interpolate, solver.right_corner),
    )
    # I think the user expects the initial time-point to be part of the grid
    # (Even though t0 is not computed by this function)
    t = jnp.concatenate((jnp.atleast_1d(t0), t))

    # I think the user expects marginals, so we compute them here
    posterior_t0, *_ = initial_condition
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


def solve_fixed_grid(
    vector_field,
    taylor_coefficients,
    grid,
    solver,
    output_scale,
):
    """Solve an initial value problem on a fixed, pre-determined grid."""
    # Initialise the Taylor series
    initial_condition = solver.solution_from_tcoeffs(
        taylor_coefficients, output_scale=output_scale
    )

    # Compute the solution
    posterior, output_scale, num_steps = _collocate.solve_fixed_grid(
        jax.tree_util.Partial(vector_field),
        *initial_condition,
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
        num_steps=num_steps,
    )


def _userfriendly_output(*, posterior, posterior_t0):
    if isinstance(posterior, markov.MarkovSeqRev):
        marginals = markov.marginals(posterior)

        marginal_t1 = jax.tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        marginals = tree_array_util.tree_append(marginals, marginal_t1)

        # We need to include the initial filtering solution (as an init)
        # Otherwise, information is lost, and we cannot, e.g., interpolate correctly.
        init_t0 = posterior_t0.init
        init = tree_array_util.tree_prepend(init_t0, posterior.init)
        posterior = markov.MarkovSeqRev(init=init, conditional=posterior.conditional)
    else:
        posterior = tree_array_util.tree_prepend(posterior_t0, posterior)
        marginals = posterior
    return marginals, posterior


R = TypeVar("R")
"""Type-variable for random variables used in \
 generic initial value problem solutions."""


@jax.tree_util.register_pytree_node_class
class Solution(Generic[R]):
    """Estimated initial value problem solution."""

    def __init__(
        self,
        t,
        u,
        output_scale,
        marginals: R,
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

    def tree_flatten(self):
        children = (
            self.t,
            self.u,
            self.marginals,
            self.posterior,
            self.output_scale,
            self.num_steps,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        t, u, marginals, posterior, output_scale, n = children
        return cls(
            t=t,
            u=u,
            marginals=marginals,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=n,
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
