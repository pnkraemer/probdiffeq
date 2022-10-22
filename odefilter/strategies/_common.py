"""Inference interface."""

import abc
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")


@jax.tree_util.register_pytree_node_class
class Solution(Generic[T]):
    """Inferred solutions."""

    def __init__(
        self,
        *,
        t: float,
        t_previous: float,
        u: float,
        output_scale_sqrtm: float,
        marginals: T,
        posterior: Any,
        num_data_points: float,
    ):
        self.t = t
        self.t_previous = t_previous
        self.u = u
        self.output_scale_sqrtm = output_scale_sqrtm
        self.marginals = marginals
        self.posterior = posterior
        self.num_data_points = num_data_points

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
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
        t, t_previous, u, marginals, posterior, output_scale_sqrtm, n = children
        return cls(
            t=t,
            t_previous=t_previous,
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
            t_previous=self.t_previous[item],
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
class Solver(abc.ABC):
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

    def init_fn(self, *, taylor_coefficients, t0):
        corrected = self.strategy.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        posterior = self.strategy.init_posterior(corrected=corrected)
        sol = self.strategy.extract_sol_terminal_value(posterior=posterior)

        solution = Solution(
            t=t0,
            t_previous=t0,
            u=sol,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=1.0,
            num_data_points=1.0,
        )

        error_estimate = self.strategy.implementation.init_error_estimate()
        return solution, error_estimate

    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102

        # Cases to switch between
        branches = [
            lambda s0_, s1_, t_: self.strategy.case_right_corner(s0=s0_, s1=s1_, t=t_),
            lambda s0_, s1_, t_: self.strategy.case_interpolate(s0=s0_, s1=s1_, t=t_),
        ]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10  # todo: magic constant?
        is_in_between = jnp.logical_not(is_right_corner)

        index_as_array, *_ = jnp.where(
            jnp.asarray([is_right_corner, is_in_between]), size=1
        )
        index = jnp.reshape(index_as_array, ())
        return jax.lax.switch(index, branches, s0, s1, t)

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
            return self.strategy.offgrid_marginals(t=t, state=s, state_previous=sprev)

        return marginals_vmap(solution_left, ts, solution_right)

    def offgrid_marginals(self, **kwargs):
        # todo: this is only temporary!! Remove soon.
        return self.strategy.offgrid_marginals(**kwargs)

    def tree_flatten(self):
        children = (self.strategy,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (strategy,) = children
        return cls(strategy=strategy)

    def _estimate_error(self, linear_fn, m_obs, p):
        scale_sqrtm, error_est = self.strategy.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_est = error_est * scale_sqrtm
        return error_est, scale_sqrtm


@jax.tree_util.register_pytree_node_class  # is this necessary?
class DynamicSolver(Solver):
    """Dynamic calibration."""

    def step_fn(self, *, state, info_op, dt, parameters):
        p, p_inv = self.strategy.implementation.assemble_preconditioner(dt=dt)

        m_ext, cache = self.strategy.extrapolate_mean(
            posterior=state.posterior, p=p, p_inv=p_inv
        )

        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)
        error_estimate, output_scale_sqrtm = self._estimate_error(linear_fn, m_obs, p)

        extrapolated = self.strategy.complete_extrapolation(
            m_ext,
            cache,
            posterior_previous=state.posterior,
            output_scale_sqrtm=output_scale_sqrtm,
            p=p,
            p_inv=p_inv,
        )

        # Final observation
        _, (corrected, _) = self.strategy.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        smoothing_solution = Solution(
            t=state.t + dt,
            t_previous=state.t,
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
            t_previous=state.t_previous,
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
            t_previous=state.t_previous,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class  # is this necessary?
class NonDynamicSolver(Solver):
    """Standard calibration. Nothing dynamic."""

    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        # Pre-error-estimate steps
        p, p_inv = self.strategy.implementation.assemble_preconditioner(dt=dt)
        m_ext, cache = self.strategy.extrapolate_mean(
            posterior=state.posterior, p_inv=p_inv, p=p
        )

        # Linearise and estimate error
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)
        error_estimate, _ = self._estimate_error(linear_fn, m_obs, p)

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            m_ext,
            cache,
            output_scale_sqrtm=1.0,
            posterior_previous=state.posterior,
            p=p,
            p_inv=p_inv,
        )

        # Complete step (incl. calibration!)
        output_scale_sqrtm, n = state.output_scale_sqrtm, state.num_data_points
        observed, (corrected, _) = self.strategy.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        new_output_scale_sqrtm = self._update_output_scale_sqrtm(
            diffsqrtm=output_scale_sqrtm, n=n, obs=observed
        )

        # Extract and return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        filtered = Solution(
            t=state.t + dt,
            t_previous=state.t,
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
            t_previous=state.t_previous,
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
            t_previous=state.t_previous,
            u=u,
            marginals=marginals,  # new!
            posterior=posterior,  # new!
            output_scale_sqrtm=s,  # new!
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC):
    """Inference strategy interface."""

    def __init__(self, *, implementation):
        self.implementation = implementation

    @abc.abstractmethod
    def init_posterior(self, *, corrected):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol_terminal_value(self, *, posterior):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol_from_marginals(self, *, marginals):
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def case_interpolate(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def marginals(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def marginals_terminal_value(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError

    @abc.abstractmethod
    def extrapolate_mean(self, *, posterior, p_inv, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, m_ext, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def final_correction(self, *, extrapolated, linear_fn, m_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)
