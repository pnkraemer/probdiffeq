"""Initial value problem solvers and solutions."""

import abc
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

RVTypeVar = TypeVar("RVTypeVar")
"""Type-variable for random variables used in \
 generic initial value problem solutions."""


@jax.tree_util.register_pytree_node_class
class Solution(Generic[RVTypeVar]):
    """Estimated initial value problem solution."""

    def __init__(
        self, t, u, output_scale_sqrtm, marginals: RVTypeVar, posterior, num_data_points
    ):
        self.t = t
        self.u = u
        self.output_scale_sqrtm = output_scale_sqrtm
        self.marginals = marginals
        self.posterior = posterior
        self.num_data_points = num_data_points

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"t={self.t},"
            f"u={self.u},"
            f"output_scale_sqrtm={self.output_scale_sqrtm},"
            f"marginals={self.marginals},"
            f"posterior={self.posterior},"
            f"num_data_points={self.num_data_points},"
            ")"
        )

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
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
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
        for i in range(self.t.shape[0]):
            yield self[i]


@jax.tree_util.register_pytree_node_class
class _AbstractSolver(abc.ABC):
    """Interface for initial value problem solvers."""

    def __init__(self, strategy):
        self.strategy = strategy

    def __eq__(self, other):
        def all_equal(a, b):
            return jnp.all(jnp.equal(a, b))

        tree_equal = jax.tree_util.tree_map(all_equal, self, other)
        return jax.tree_util.tree_all(tree_equal)

    def __repr__(self):
        return f"{self.__class__.__name__}(strategy={self.strategy})"

    # Abstract methods

    @abc.abstractmethod
    def step_fn(self, *, state, vector_field, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, *, state):
        raise NotImplementedError

    def init_fn(self, *, taylor_coefficients, t0):
        posterior = self.strategy.init_posterior(
            taylor_coefficients=taylor_coefficients
        )
        sol = self.strategy.extract_sol_terminal_value(posterior=posterior)

        scale_sqrtm = self.strategy.init_output_scale_sqrtm()
        error_estimate = self.strategy.init_error_estimate()

        solution = Solution(
            t=t0,
            u=sol,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=scale_sqrtm,
            num_data_points=1.0,
        )

        return solution, error_estimate

    def interpolate_fn(self, *, s0, s1, t):
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

    def tree_flatten(self):
        children = (self.strategy,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (strategy,) = children
        return cls(strategy=strategy)


@jax.tree_util.register_pytree_node_class
class CalibrationFreeSolver(_AbstractSolver):
    """Initial value problem solver.

    No automatic output-scale calibration.
    """

    def __init__(self, strategy, *, output_scale_sqrtm):
        super().__init__(strategy=strategy)

        # todo: overwrite init_fn()?
        self._output_scale_sqrtm = output_scale_sqrtm

    def step_fn(self, *, state, vector_field, dt, parameters):
        # Pre-error-estimate steps
        linearisation_pt, cache_ext = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            cache_ext,
            output_scale_sqrtm=self._output_scale_sqrtm,  # todo: use from state?
            posterior_previous=state.posterior,
        )

        # Complete step (incl. calibration!)
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated,
            cache_obs=cache_obs,
        )

        # Extract and return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        filtered = Solution(
            t=state.t + dt,
            u=sol,
            marginals=None,
            posterior=corrected,
            output_scale_sqrtm=self._output_scale_sqrtm,  # todo: use from state?
            num_data_points=state.num_data_points + 1,
        )
        return filtered, dt * error

    # todo: move this to the abstract solver and overwrite when necessary?
    #  the dynamic solver uses the same...
    def extract_fn(self, *, state):
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()
        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()
        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def tree_flatten(self):
        children = (self.strategy, self._output_scale_sqrtm)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (strategy, output_scale_sqrtm) = children
        return cls(strategy=strategy, output_scale_sqrtm=output_scale_sqrtm)


@jax.tree_util.register_pytree_node_class
class DynamicSolver(_AbstractSolver):
    """Initial value problem solver with dynamic calibration of the output scale."""

    def step_fn(self, *, state, vector_field, dt, parameters):
        linearisation_pt, cache_ext = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )
        error, scale_sqrtm, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            cache_ext,
            posterior_previous=state.posterior,
            output_scale_sqrtm=scale_sqrtm,
        )

        # Final observation
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated, cache_obs=cache_obs
        )

        # Return solution
        sol = self.strategy.extract_sol_terminal_value(posterior=corrected)
        smoothing_solution = Solution(
            t=state.t + dt,
            u=sol,
            posterior=corrected,
            marginals=None,
            output_scale_sqrtm=scale_sqrtm,
            num_data_points=state.num_data_points + 1,
        )

        return smoothing_solution, dt * error

    def extract_fn(self, *, state):
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()

        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()

        return Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class MLESolver(_AbstractSolver):
    """Initial value problem solver with (quasi-)maximum-likelihood \
     calibration of the output-scale."""

    def step_fn(self, *, state, vector_field, dt, parameters):
        # Pre-error-estimate steps
        linearisation_pt, cache_ext = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            cache_ext,
            output_scale_sqrtm=self.strategy.init_output_scale_sqrtm(),
            posterior_previous=state.posterior,
        )
        # Complete step (incl. calibration!)
        output_scale_sqrtm, n = state.output_scale_sqrtm, state.num_data_points
        observed, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated,
            cache_obs=cache_obs,
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
        return filtered, dt * error

    @staticmethod
    def _update_output_scale_sqrtm(*, diffsqrtm, n, obs):
        x = obs.norm_of_whitened_residual_sqrtm()
        return jnp.sqrt(n * diffsqrtm**2 + x**2) / jnp.sqrt(n + 1)

    def extract_fn(self, *, state):
        s = state.output_scale_sqrtm[-1] * jnp.ones_like(state.output_scale_sqrtm)
        margs = self.strategy.marginals(posterior=state.posterior)
        return self._rescale(scale_sqrtm=s, marginals_unscaled=margs, state=state)

    def extract_terminal_value_fn(self, *, state):
        s = state.output_scale_sqrtm
        margs = self.strategy.marginals_terminal_value(posterior=state.posterior)
        return self._rescale(scale_sqrtm=s, marginals_unscaled=margs, state=state)

    @staticmethod
    def _rescale(*, scale_sqrtm, marginals_unscaled, state):
        # todo: these calls to *.scale_covariance are a bit cumbersome,
        #  because we need to add this
        #  method to all sorts of classes.
        #  Instead, we could move the collections.AbstractNormal
        #  to the top-level and implement this via tree_map:
        #  def scale_cov(tree):
        #      def is_leaf(x):
        #          return isinstance(x, AbstractNormal)
        #      def fn(x: AbstractNormal):
        #          return x.scale_covariance(scale_sqrtm)
        #      return tree_map(fn, tree, is_leaf=is_leaf)
        #  marginals = scale_cov(marginals_unscaled)
        #  posterior = scale_cov(state.posterior)
        #  Which would avoid having to do this over and over again
        #  in intermediate objects
        #  (Conditionals, Posteriors, StateSpaceVars, ...)

        marginals = marginals_unscaled.scale_covariance(scale_sqrtm=scale_sqrtm)
        posterior = state.posterior.scale_covariance(scale_sqrtm=scale_sqrtm)
        u = marginals.extract_qoi()
        return Solution(
            t=state.t,
            u=u,
            marginals=marginals,  # new!
            posterior=posterior,  # new!
            output_scale_sqrtm=scale_sqrtm,  # new!
            num_data_points=state.num_data_points,
        )
