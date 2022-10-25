"""Inference interface."""

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")

# todo: no method in here should call self.strategy.implementation.


@jax.tree_util.register_pytree_node_class
@dataclass
class Solution(Generic[T]):
    """Inferred solutions."""

    t: float
    u: float
    output_scale_sqrtm: float
    marginals: T
    posterior: Any
    num_data_points: float

    def tree_flatten(self):
        children = (
            self.t,
            self.u,
            self.marginals,
            self.posterior,
            self.output_scale_sqrtm,
            self.num_data_points,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        t, u, marginals, posterior, output_scale_sqrtm, n = children
        return cls(
            t=t,
            u=u,
            marginals=marginals,
            posterior=posterior,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=n,
        )

    def __len__(self):
        """Length of a solution object.

        Depends on the length of the underlying :attr:`t` attribute.
        """
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access the `i`-th sub-solution."""
        if jnp.ndim(self.t) < 1:
            raise ValueError(f"Solution object not batched :(, {jnp.ndim(self.t)}")
        if isinstance(item, tuple) and len(item) > jnp.ndim(self.t):
            # s[2, 3] forbidden
            raise ValueError(f"Inapplicable shape: {item, jnp.shape(self.t)}")
        return Solution(
            t=self.t[item],
            u=self.u[item],
            output_scale_sqrtm=self.output_scale_sqrtm[item],
            # todo: make iterable?
            marginals=jax.tree_util.tree_map(lambda x: x[item], self.marginals),
            # todo: make iterable?
            posterior=jax.tree_util.tree_map(lambda x: x[item], self.posterior),
            num_data_points=self.num_data_points[item],
        )

    def __iter__(self):
        """Iterate through the filtering solution."""
        for i in range(self.t.shape[0]):
            yield self[i]


@jax.tree_util.register_pytree_node_class
class _Solver(abc.ABC):
    """Inference strategy interface."""

    def __init__(self, *, strategy):
        self.strategy = strategy

    # Abstract methods

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, *, state):
        raise NotImplementedError

    def sample(self, key, *, solution, shape=()):
        return self.strategy.sample(key, posterior=solution.posterior, shape=shape)

    def init_fn(self, *, taylor_coefficients, t0):
        corrected = self.strategy.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        posterior = self.strategy.init_posterior(corrected=corrected)
        sol = self.strategy.extract_sol_terminal_value(posterior=posterior)
        scale_sqrtm = self.strategy.implementation.init_output_scale_sqrtm()
        solution = Solution(
            t=t0,
            u=sol,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=scale_sqrtm,
            num_data_points=1.0,
        )

        error_estimate = self.strategy.implementation.init_error_estimate()
        return solution, error_estimate

    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102
        def interpolate(s0_, s1_, t_):
            return self.strategy.case_interpolate(
                p0=s0_.posterior,
                rv1=self.strategy.marginals_terminal_value(posterior=s1.posterior),
                t=t_,
                t0=s0_.t,
                t1=s1_.t,
                scale_sqrtm=s1.output_scale_sqrtm,
            )

        def right_corner(s0_, s1_, t_):
            # todo: are all these arguments needed?
            return self.strategy.case_right_corner(
                p0=s0_.posterior,
                p1=s1_.posterior,
                t=t_,
                t0=s0_.t,
                t1=s1_.t,
                scale_sqrtm=s1.output_scale_sqrtm,
            )

        # Cases to switch between
        branches = [right_corner, interpolate]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10  # todo: magic constant?
        is_in_between = jnp.logical_not(is_right_corner)

        index_as_array, *_ = jnp.where(
            jnp.asarray([is_right_corner, is_in_between]), size=1
        )
        index = jnp.reshape(index_as_array, ())
        acc, sol, prev = jax.lax.switch(index, branches, s0, s1, t)

        previous = self._posterior_to_state(prev, t, s1)
        solution = self._posterior_to_state(sol, t, s1)
        accepted = self._posterior_to_state(acc, jnp.maximum(s1.t, t), s1)

        return accepted, solution, previous

    def _posterior_to_state(self, posterior, t, state):
        return Solution(
            t=t,
            u=self.strategy.extract_sol_terminal_value(posterior=posterior),
            posterior=posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            marginals=None,
            num_data_points=state.num_data_points,
        )

    def offgrid_marginals_searchsorted(self, *, ts, solution):
        """Dense output for a whole grid via jax.numpy.searchsorted.

        !!! warning
            The elements in ts and the elements in the solution grid must be disjoint.
            Otherwise, anything can happen and the solution will be incorrect.
            We do not check for this case! (Because we want to jit!)

        !!! warning
            The elements in ts must be strictly in (t0, t1).
            Again there is no check and anything can happen if you don't follow
            this rule.
        """
        # todo: support "method" argument.

        # side="left" and side="right" are equivalent
        # because we _assume_ that the point sets are disjoint.
        indices = jnp.searchsorted(solution.t, ts)

        # Solution slicing to the rescue
        solution_left = solution[indices - 1]
        solution_right = solution[indices]

        # Vmap to the rescue :) It does not like kw-only arguments, though.
        @jax.vmap
        def marginals_vmap(sprev, t, s):
            return self.offgrid_marginals(t=t, state=s, state_previous=sprev)

        return marginals_vmap(solution_left, ts, solution_right)

    def offgrid_marginals(self, *, state, t, state_previous):
        return self.strategy.offgrid_marginals(
            marginals=state.marginals,
            posterior_previous=state_previous.posterior,
            t=t,
            t0=state_previous.t,
            t1=state.t,
            scale_sqrtm=state.output_scale_sqrtm,
        )

    def tree_flatten(self):
        children = (self.strategy,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (strategy,) = children
        return cls(strategy=strategy)

    def _estimate_error(self, *, info_op, cache_obs, m_obs, p):

        # todo: one sho
        scale_sqrtm, error_est = self.strategy.implementation.estimate_error(
            info_op=info_op, cache_obs=cache_obs, m_obs=m_obs, p=p
        )
        error_est = error_est * scale_sqrtm
        return error_est, scale_sqrtm


@jax.tree_util.register_pytree_node_class  # is this necessary?
class DynamicSolver(_Solver):
    """Dynamic calibration."""

    def step_fn(self, *, state, info_op, dt, parameters):
        p, p_inv = self.strategy.implementation.assemble_preconditioner(dt=dt)

        m_ext, cache_ext = self.strategy.extrapolate_mean(
            posterior=state.posterior, p=p, p_inv=p_inv
        )

        m_obs, cache_obs = info_op.linearize(m_ext, t=state.t + dt, p=parameters)
        error_estimate, output_scale_sqrtm = self._estimate_error(
            info_op=info_op, cache_obs=cache_obs, m_obs=m_obs, p=p
        )

        extrapolated = self.strategy.complete_extrapolation(
            m_ext,
            cache_ext,
            posterior_previous=state.posterior,
            output_scale_sqrtm=output_scale_sqrtm,
            p=p,
            p_inv=p_inv,
        )

        # Final observation
        _, (corrected, _) = self.strategy.final_correction(
            info_op=info_op, extrapolated=extrapolated, cache_obs=cache_obs, m_obs=m_obs
        )

        # Return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        smoothing_solution = Solution(
            t=state.t + dt,
            u=sol,
            posterior=corrected,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=state.num_data_points + 1,
        )

        return smoothing_solution, dt * error_estimate

    def extract_fn(self, *, state):  # noqa: D102

        marginals = self.strategy.marginals(posterior=state.posterior)
        u = self.strategy.extract_sol_from_marginals(marginals=marginals)

        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):  # noqa: D102
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = self.strategy.extract_sol_from_marginals(marginals=marginals)

        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class  # is this necessary?
class NonDynamicSolver(_Solver):
    """Standard calibration. Nothing dynamic."""

    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        # Pre-error-estimate steps
        p, p_inv = self.strategy.implementation.assemble_preconditioner(dt=dt)
        m_ext, cache_ext = self.strategy.extrapolate_mean(
            posterior=state.posterior, p_inv=p_inv, p=p
        )

        # Linearise and estimate error
        m_obs, cache_obs = info_op.linearize(m_ext, t=state.t + dt, p=parameters)
        error_estimate, _ = self._estimate_error(
            info_op=info_op, cache_obs=cache_obs, m_obs=m_obs, p=p
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            m_ext,
            cache_ext,
            output_scale_sqrtm=self.strategy.implementation.init_output_scale_sqrtm(),
            posterior_previous=state.posterior,
            p=p,
            p_inv=p_inv,
        )

        # Complete step (incl. calibration!)
        output_scale_sqrtm, n = state.output_scale_sqrtm, state.num_data_points
        observed, (corrected, _) = self.strategy.final_correction(
            info_op=info_op, extrapolated=extrapolated, cache_obs=cache_obs, m_obs=m_obs
        )
        new_output_scale_sqrtm = self._update_output_scale_sqrtm(
            diffsqrtm=output_scale_sqrtm, n=n, obs=observed
        )

        # Extract and return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        filtered = Solution(
            t=state.t + dt,
            u=sol,
            marginals=None,
            posterior=corrected,
            output_scale_sqrtm=new_output_scale_sqrtm,
            num_data_points=n + 1,
        )
        return filtered, dt * error_estimate

    def _update_output_scale_sqrtm(self, *, diffsqrtm, n, obs):
        evidence_sqrtm = self.strategy.implementation.evidence_sqrtm(observed=obs)
        return jnp.sqrt(n * diffsqrtm**2 + evidence_sqrtm**2) / jnp.sqrt(n + 1)

    def extract_fn(self, *, state):  # noqa: D102

        marginals = self.strategy.marginals(posterior=state.posterior)
        s = state.output_scale_sqrtm[-1] * jnp.ones_like(state.output_scale_sqrtm)

        marginals = self.strategy.scale_marginals(marginals, output_scale_sqrtm=s)
        posterior = self.strategy.scale_posterior(state.posterior, output_scale_sqrtm=s)

        u = self.strategy.extract_sol_from_marginals(marginals=marginals)
        return Solution(
            t=state.t,
            u=u,
            marginals=marginals,  # new!
            posterior=posterior,  # new!
            output_scale_sqrtm=s,  # new!
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        s = state.output_scale_sqrtm

        marginals = self.strategy.scale_marginals(marginals, output_scale_sqrtm=s)
        posterior = self.strategy.scale_posterior(state.posterior, output_scale_sqrtm=s)

        u = self.strategy.extract_sol_from_marginals(marginals=marginals)
        return Solution(
            t=state.t,
            u=u,
            marginals=marginals,  # new!
            posterior=posterior,  # new!
            output_scale_sqrtm=s,  # new!
            num_data_points=state.num_data_points,
        )
