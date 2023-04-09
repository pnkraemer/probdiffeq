"""Calibrated IVP solvers."""

import abc
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util, solution


class _State(NamedTuple):
    """Solver state."""

    # Same as in solution.Solution()
    t: Any
    u: Any
    posterior: Any
    num_data_points: Any

    # Not contained in _State but in Solution: output_scale, marginals.

    # Different to solution.Solution():
    error_estimate: Any
    output_scale_calibrated: Any
    output_scale_prior: Any


@jax.tree_util.register_pytree_node_class
class AbstractSolver(abc.ABC):
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
    def step_fn(
        self, *, state: _State, vector_field, dt, parameters, output_scale
    ) -> _State:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    # # todo: change to empty_solution_from_tcoeffs?
    # def posterior_from_tcoeffs(self, taylor_coefficients, /):
    #     posterior = self.strategy.init_posterior(
    #         taylor_coefficients=taylor_coefficients
    #     )
    #     return posterior

    def empty_solution_from_tcoeffs(self, taylor_coefficients, /, **kwargs):
        """Construct an initial `Solution` object.

        An (even if empty) solution object is needed to initialise the solver.
        Thus, this method is kind-of a helper function to make the rest of the
        initialisation code a bit simpler.
        """
        posterior = self.strategy.init_posterior(
            taylor_coefficients=taylor_coefficients
        )
        u = taylor_coefficients[0]
        return self.empty_solution_from_posterior(posterior, u=u, **kwargs)

    def empty_solution_from_posterior(self, posterior, /, *, u, t, output_scale):
        output_scale = self.strategy.init_output_scale(output_scale)
        return solution.Solution(
            t=t,
            posterior=posterior,
            marginals=self.strategy.marginals_terminal_value(posterior),
            output_scale=output_scale,
            u=u,
            num_data_points=1.0,
        )

    def init(self, sol, /) -> _State:
        # todo: if we `init()` this output scale, should we also `extract()`?
        error_estimate = self.strategy.init_error_estimate()

        # discard sol.marginals. Add an error estimate instead.
        return _State(
            t=sol.t,
            u=sol.u,
            error_estimate=error_estimate,
            posterior=sol.posterior,
            output_scale_prior=sol.output_scale,
            output_scale_calibrated=sol.output_scale,
            num_data_points=sol.num_data_points,
        )

    def interpolate_fn(self, *, s0: _State, s1: _State, t):
        raise RuntimeError(
            "Next up: wrap strategy.case_interpolate and case_right_corner "
            "into solver methods that operate on solver states. "
            "Currently, we lose too much information and need to use functions "
            "such as make_state, which should absolutely not be necessary."
            "Once this is done, make dynamic solver use local_scale as BOTH output scales, "
            "and keep fixing all the tests (we are in the middle of state having "
            "two different output scales). Then, remove output_scale from step_fn() "
            "and keep fixing all failing tests. Once this is done, "
            "we should be ready to look at the Pull request diff "
            "(as we are done splitting _State from Solution() -- and "
            "maybe even done with extract(init(solver))?"
        )

        def interpolate(s0_: _State, s1_: _State, t_):
            return self.strategy.case_interpolate(
                p0=s0_.posterior,
                rv1=self.strategy.marginals_terminal_value(posterior=s1.posterior),
                t=t_,
                t0=s0_.t,
                t1=s1_.t,
                output_scale=s1.output_scale_prior,
            )

        def right_corner(s0_: _State, s1_: _State, t_):
            # todo: are all these arguments needed?
            return self.strategy.case_right_corner(
                p0=s0_.posterior,
                p1=s1_.posterior,
                t=t_,
                t0=s0_.t,
                t1=s1_.t,
                output_scale=s1.output_scale_prior,
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

        # helper function to make code below more readable
        def make_state(p, t_) -> _State:
            return self.init(
                posterior=p,
                t=t_,
                u=self.strategy.extract_u_from_posterior(p),
                output_scale_calibrated=s1.output_scale,
                output_scale=s1.output_scale,
                num_data_points=s1.num_data_points,
            )

        # todo: which output scale is used for MLESolver interpolation
        #  _during_ the simulation? hopefully the prior one!

        previous = make_state(prev, t)
        solution_ = make_state(sol, t)
        accepted = make_state(acc, jnp.maximum(s1.t, t))

        return accepted, solution_, previous

    def tree_flatten(self):
        children = (self.strategy,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (strategy,) = children
        return cls(strategy=strategy)


@jax.tree_util.register_pytree_node_class
class CalibrationFreeSolver(AbstractSolver):
    """Initial value problem solver.

    No automatic output-scale calibration.
    """

    def step_fn(
        self, *, state: _State, vector_field, dt, parameters, output_scale
    ) -> _State:
        # Pre-error-estimate steps
        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        scale = self.strategy.init_output_scale(output_scale)
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            output_scale=scale,
            posterior_previous=state.posterior,
        )

        # Complete step (incl. calibration!)
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated,
            cache_obs=cache_obs,
        )

        # Extract and return solution
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        return _State(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            output_scale=scale,
            num_data_points=state.num_data_points + 1,
        )

    # todo: move this to the abstract solver and overwrite when necessary?
    #  the dynamic solver uses the same...
    def extract_fn(self, state: _State, /) -> solution.Solution:
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class DynamicSolver(AbstractSolver):
    """Initial value problem solver with dynamic calibration of the output scale."""

    def step_fn(
        self, *, state: _State, vector_field, dt, parameters, output_scale
    ) -> _State:
        del output_scale  # unused

        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )
        error, output_scale, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            posterior_previous=state.posterior,
            output_scale=output_scale,
        )

        # Final observation
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated, cache_obs=cache_obs
        )

        # Return solution
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        return _State(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            output_scale=output_scale,
            num_data_points=state.num_data_points + 1,
        )

    def extract_fn(self, state: _State, /) -> solution.Solution:
        marginals = self.strategy.marginals(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class MLESolver(AbstractSolver):
    """Initial value problem solver with (quasi-)maximum-likelihood \
     calibration of the output-scale."""

    def step_fn(
        self, *, state: _State, vector_field, dt, parameters, output_scale
    ) -> _State:
        # Pre-error-estimate steps
        linearisation_pt = self.strategy.begin_extrapolation(
            posterior=state.posterior, dt=dt
        )

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            linearisation_pt, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        scale = self.strategy.init_output_scale(output_scale)
        extrapolated = self.strategy.complete_extrapolation(
            linearisation_pt,
            output_scale=scale,
            posterior_previous=state.posterior,
        )
        # Complete step (incl. calibration!)
        observed, (corrected, _) = self.strategy.complete_correction(
            extrapolated=extrapolated,
            cache_obs=cache_obs,
        )
        output_scale, n = state.output_scale_calibrated, state.num_data_points
        new_output_scale = self._update_output_scale(
            diffsqrtm=output_scale, n=n, obs=observed
        )
        # todo: remove output_scale from step_fn() signature.

        # Extract and return solution
        u = self.strategy.extract_u_from_posterior(posterior=corrected)
        return _State(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            output_scale_prior=state.output_scale_prior,
            output_scale_calibrated=new_output_scale,
            num_data_points=n + 1,
        )

    @staticmethod
    def _update_output_scale(*, diffsqrtm, n, obs):
        # Special consideration for block-diagonal models
        # todo: should this logic be pushed to the implementation itself?
        if jnp.ndim(diffsqrtm) > 0:

            def fn_partial(d, o):
                fn = MLESolver._update_output_scale
                return fn(diffsqrtm=d, n=n, obs=o)

            fn_vmap = jax.vmap(fn_partial)
            return fn_vmap(diffsqrtm, obs)

        x = obs.mahalanobis_norm(jnp.zeros_like(obs.mean)) / jnp.sqrt(obs.mean.size)
        sum = _sqrt_util.sqrt_sum_square(jnp.sqrt(n) * diffsqrtm, x)
        return sum / jnp.sqrt(n + 1)

    def extract_fn(self, state: _State, /) -> solution.Solution:
        s = state.output_scale[-1] * jnp.ones_like(state.output_scale)
        marginals = self.strategy.marginals(posterior=state.posterior)
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=state.t,
            u=state.u,
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        s = state.output_scale
        marginals = self.strategy.marginals_terminal_value(posterior=state.posterior)
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=state.t,
            u=state.u,
            marginals=marginals,  # new!
            posterior=state.posterior,
            output_scale=state.output_scale,
            num_data_points=state.num_data_points,
        )

    @staticmethod
    def _rescale_covs(state, /, *, output_scale, marginals_unscaled):
        # todo: these calls to *.scale_covariance are a bit cumbersome,
        #  because we need to add this
        #  method to all sorts of classes.
        #  Instead, we could move the collections.AbstractNormal
        #  to the top-level and implement this via tree_map:
        #  def scale_cov(tree):
        #      def is_leaf(x):
        #          return isinstance(x, AbstractNormal)
        #      def fn(x: AbstractNormal):
        #          return x.scale_covariance(output_scale)
        #      return tree_map(fn, tree, is_leaf=is_leaf)
        #  marginals = scale_cov(marginals_unscaled)
        #  posterior = scale_cov(state.posterior)
        #  Which would avoid having to do this over and over again
        #  in intermediate objects
        #  (Conditionals, Posteriors, StateSpaceVars, ...)

        marginals = marginals_unscaled.scale_covariance(output_scale)
        posterior = state.posterior.scale_covariance(output_scale)
        u = marginals.extract_qoi()
        return _State(
            t=state.t,
            u=u,
            posterior=posterior,
            output_scale=output_scale,
            error_estimate=None,  # irrelevant, will be removed in next step.
            num_data_points=state.num_data_points,
        )
