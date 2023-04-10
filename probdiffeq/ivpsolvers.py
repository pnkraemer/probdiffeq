"""Calibrated IVP solvers."""

import abc
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probdiffeq import _collections, _sqrt_util, solution


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
    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    def solution_from_tcoeffs(self, taylor_coefficients, /, **kwargs):
        """Construct an initial `Solution` object.

        An (even if empty) solution object is needed to initialise the solver.
        Thus, this method is kind-of a helper function to make the rest of the
        initialisation code a bit simpler.
        """
        posterior = self.strategy.init(taylor_coefficients=taylor_coefficients)
        u = taylor_coefficients[0]
        return self.solution_from_posterior(posterior, u=u, **kwargs)

    def solution_from_posterior(self, marginals, posterior, /, *, u, t, output_scale):
        """Use for initialisation but also for interpolation."""
        # todo: if we `init()` this output scale, should we also `extract()`?
        output_scale = self.strategy.init_output_scale(output_scale)

        # todo: make "marginals" an input to this function,
        #  then couple it with the posterior
        #  and make (u, marginals, posterior) the "solution" type for strategies.
        marginals = self.strategy.extract_marginals_terminal_values(posterior)
        return solution.Solution(
            t=t,
            posterior=posterior,
            marginals=marginals,
            output_scale=output_scale,
            u=u,
            num_data_points=1.0,
        )

    def init(self, sol, /) -> _State:
        error_estimate = self.strategy.init_error_estimate()
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
        # Cases to switch between
        branches = [self.case_right_corner, self.case_interpolate]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10  # todo: magic constant?
        is_in_between = jnp.logical_not(is_right_corner)
        index_as_array = jnp.asarray([is_right_corner, is_in_between])

        # Select branch and return result
        apply_branch_as_array, *_ = jnp.where(index_as_array, size=1)
        apply_branch = jnp.reshape(apply_branch_as_array, ())
        return jax.lax.switch(apply_branch, branches, s0, s1, t)

    def case_interpolate(
        self, s0: _State, s1: _State, t
    ) -> _collections.InterpRes[_State]:
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            p0=s0.posterior,
            p1=s1.posterior,
            t=t,
            t0=s0.t,
            t1=s1.t,
            # always interpolate with the prior output scale.
            #  This is important to make the MLE solver behave correctly.
            #  (Dynamic solvers overwrite the prior output scale at every step anyway).
            output_scale=s1.output_scale_prior,
        )
        t_accepted = jnp.maximum(s1.t, t)
        prev = self._interp_make_state(prev_p, t=t, reference=s0)
        sol = self._interp_make_state(sol_p, t=t, reference=s1)
        acc = self._interp_make_state(acc_p, t=t_accepted, reference=s1)
        return _collections.InterpRes(accepted=acc, solution=sol, previous=prev)

    def case_right_corner(
        self, s0: _State, s1: _State, t
    ) -> _collections.InterpRes[_State]:
        # todo: are all these arguments needed?
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(
            p0=s0.posterior,
            p1=s1.posterior,
            t=t,
            t0=s0.t,
            t1=s1.t,
            output_scale=s1.output_scale_prior,
        )
        t_accepted = jnp.maximum(s1.t, t)
        prev = self._interp_make_state(prev_p, t=t, reference=s0)
        sol = self._interp_make_state(sol_p, t=t, reference=s1)
        acc = self._interp_make_state(acc_p, t=t_accepted, reference=s1)
        return _collections.InterpRes(accepted=acc, solution=sol, previous=prev)

    def _interp_make_state(self, posterior, *, t, reference: _State) -> _State:
        error_estimate = self.strategy.init_error_estimate()
        u = self.strategy.extract_u(posterior)
        return _State(
            posterior=posterior,
            t=t,
            u=u,
            num_data_points=reference.num_data_points,
            error_estimate=error_estimate,
            output_scale_prior=reference.output_scale_prior,
            output_scale_calibrated=reference.output_scale_calibrated,
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
class CalibrationFreeSolver(AbstractSolver):
    """Initial value problem solver.

    No automatic output-scale calibration.
    """

    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        # Pre-error-estimate steps
        output_extra = self.strategy.begin_extrapolation(state.posterior, dt=dt)

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            output_extra, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            output_extra,
            output_scale=state.output_scale_prior,
            posterior_previous=state.posterior,
        )

        # Complete step (incl. calibration!)
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated,
            cache_obs=cache_obs,
        )

        # Extract and return solution
        u = self.strategy.extract_u(corrected)
        return _State(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            output_scale_prior=state.output_scale_prior,
            # Nothing happens in the field below:
            #  but we cannot use "None" if we want to reuse the init()
            #  method from abstract solvers (which populate this field).
            output_scale_calibrated=state.output_scale_prior,
            num_data_points=state.num_data_points + 1,
        )

    def extract_fn(self, state: _State, /) -> solution.Solution:
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            # _prior and _calibrated are identical.
            #  but we use _prior because we might remove the _calibrated
            #  value in the future.
            output_scale=state.output_scale_prior,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_prior,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class DynamicSolver(AbstractSolver):
    """Initial value problem solver with dynamic calibration of the output scale."""

    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        output_extra = self.strategy.begin_extrapolation(state.posterior, dt=dt)
        error, output_scale, cache_obs = self.strategy.begin_correction(
            output_extra, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        extrapolated = self.strategy.complete_extrapolation(
            output_extra,
            posterior_previous=state.posterior,
            output_scale=output_scale,
        )

        # Final observation
        _, (corrected, _) = self.strategy.complete_correction(
            extrapolated, cache_obs=cache_obs
        )

        # Return solution
        u = self.strategy.extract_u(corrected)
        return _State(
            t=state.t + dt,
            u=u,
            error_estimate=dt * error,
            posterior=corrected,
            output_scale_calibrated=output_scale,
            # current scale becomes the new prior scale!
            #  this is because dynamic solvers assume a piecewise-constant model
            output_scale_prior=output_scale,
            num_data_points=state.num_data_points + 1,
        )

    def extract_fn(self, state: _State, /) -> solution.Solution:
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=state.t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=state.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class MLESolver(AbstractSolver):
    """Initial value problem solver with (quasi-)maximum-likelihood \
     calibration of the output-scale."""

    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        # Pre-error-estimate steps
        output_extra = self.strategy.begin_extrapolation(state.posterior, dt=dt)

        # Linearise and estimate error
        error, _, cache_obs = self.strategy.begin_correction(
            output_extra, vector_field=vector_field, t=state.t + dt, p=parameters
        )

        # Post-error-estimate steps
        extrapolated = self.strategy.complete_extrapolation(
            output_extra,
            output_scale=state.output_scale_prior,
            posterior_previous=state.posterior,
        )
        # Complete step (incl. calibration!)
        observed, (corrected, _) = self.strategy.complete_correction(
            extrapolated,
            cache_obs=cache_obs,
        )
        output_scale, n = state.output_scale_calibrated, state.num_data_points
        new_output_scale = self._update_output_scale(
            diffsqrtm=output_scale, n=n, obs=observed
        )

        # Extract and return solution
        u = self.strategy.extract_u(corrected)
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
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals(posterior)

        # promote calibrated scale to the correct batch-shape
        s = state.output_scale_calibrated[-1] * jnp.ones_like(state.output_scale_prior)
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=state.t,
            u=state.u,
            marginals=marginals,
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, state: _State, /) -> solution.Solution:
        posterior = self.strategy.extract(state.posterior)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)

        s = state.output_scale_calibrated
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=state.t,
            u=state.u,
            marginals=marginals,
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
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
            output_scale_calibrated=output_scale,
            output_scale_prior=None,  # irrelevant, will be removed in next step
            error_estimate=None,  # irrelevant, will be removed in next step.
            num_data_points=state.num_data_points,
        )
