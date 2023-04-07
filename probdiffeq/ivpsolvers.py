"""Calibrated IVP solvers."""

import abc

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util, solution


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
        u = self.strategy.extract_u_from_posterior(posterior=posterior)

        scale_sqrtm = self.strategy.init_output_scale_sqrtm()
        error_estimate = self.strategy.init_error_estimate()

        sol = solution.Solution(
            t=t0,
            u=u,
            error_estimate=error_estimate,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=scale_sqrtm,
            num_data_points=1.0,
        )

        return sol

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

        error_estimate = jnp.empty_like(s0.error_estimate)
        previous = self._posterior_to_state(prev, t, s1, error_estimate)
        solution = self._posterior_to_state(sol, t, s1, error_estimate)
        accepted = self._posterior_to_state(
            acc, jnp.maximum(s1.t, t), s1, error_estimate
        )

        return accepted, solution, previous

    def _posterior_to_state(self, posterior, t, state, error_estimate):
        return solution.Solution(
            t=t,
            u=self.strategy.extract_u_from_posterior(posterior=posterior),
            error_estimate=error_estimate,
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

    def __repr__(self):
        name = self.__class__.__name__
        args = (
            f"strategy={self.strategy}, output_scale_sqrtm={self._output_scale_sqrtm}"
        )
        return f"{name}({args})"

    def step_fn(self, *, state, vector_field, dt, parameters):
        # Pre-error-estimate steps
        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            output_scale_sqrtm=self._output_scale_sqrtm,  # todo: use from state?
            posterior_previous=state.posterior,
        )

        # Complete step (incl. calibration!)
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated,
            cache_obs=cache_obs,
        )

        # Extract and return solution
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        filtered = solution.Solution(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            marginals=None,
            posterior=corrected,
            output_scale_sqrtm=self._output_scale_sqrtm,  # todo: use from state?
            num_data_points=state.num_data_points + 1,
        )
        return filtered

    # todo: move this to the abstract solver and overwrite when necessary?
    #  the dynamic solver uses the same...
    def extract_fn(self, *, state):
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            # error estimate is now irrelevant
            error_estimate=jnp.empty_like(state.error_estimate),
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            # error estimate is now irrelevant
            error_estimate=jnp.empty_like(state.error_estimate),
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
        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )
        error, scale_sqrtm, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            posterior_previous=state.posterior,
            output_scale_sqrtm=scale_sqrtm,
        )

        # Final observation
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated, cache_obs=cache_obs
        )

        # Return solution
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        smoothing_solution = solution.Solution(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            marginals=None,
            output_scale_sqrtm=scale_sqrtm,
            num_data_points=state.num_data_points + 1,
        )

        return smoothing_solution

    def extract_fn(self, *, state):
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()

        return solution.Solution(
            t=state.t,
            u=u,  # new!
            # error estimate is now irrelevant
            error_estimate=jnp.empty_like(state.error_estimate),
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()

        return solution.Solution(
            t=state.t,
            u=u,  # new!
            # error estimate is now irrelevant
            error_estimate=jnp.empty_like(state.error_estimate),
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
        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
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
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        filtered = solution.Solution(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            marginals=None,
            posterior=corrected,
            output_scale_sqrtm=new_output_scale_sqrtm,
            num_data_points=n + 1,
        )
        return filtered

    @staticmethod
    def _update_output_scale_sqrtm(*, diffsqrtm, n, obs):
        # Special consideration for block-diagonal models
        # todo: should this logic be pushed to the implementation itself?
        if jnp.ndim(diffsqrtm) > 0:

            def fn_partial(d, o):
                fn = MLESolver._update_output_scale_sqrtm
                return fn(diffsqrtm=d, n=n, obs=o)

            fn_vmap = jax.vmap(fn_partial)
            return fn_vmap(diffsqrtm, obs)

        x = obs.mahalanobis_norm(jnp.zeros_like(obs.mean)) / jnp.sqrt(obs.mean.size)
        sum = _sqrt_util.sqrt_sum_square(jnp.sqrt(n) * diffsqrtm, x)
        return sum / jnp.sqrt(n + 1)

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
        return solution.Solution(
            t=state.t,
            u=u,
            # error estimate is now irrelevant
            error_estimate=jnp.empty_like(state.error_estimate),
            marginals=marginals,  # new!
            posterior=posterior,  # new!
            output_scale_sqrtm=scale_sqrtm,  # new!
            num_data_points=state.num_data_points,
        )
