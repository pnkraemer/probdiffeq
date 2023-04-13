"""Calibrated IVP solvers."""

import abc
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util, solution
from probdiffeq._collections import InterpRes  # simplify type signatures


class _State(NamedTuple):
    """Solver state."""

    # Same as in solution.Solution()
    u: Any
    strategy: Any

    # Not contained in _State but in Solution: output_scale, marginals.

    # Different to solution.Solution():
    error_estimate: Any
    output_scale_calibrated: Any
    output_scale_prior: Any

    @property
    def t(self):
        return self.strategy.t


@jax.tree_util.register_pytree_node_class
class AbstractSolver(abc.ABC):
    """Interface for initial value problem solvers."""

    def __init__(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return f"{self.__class__.__name__}({self.strategy})"

    # Abstract methods

    @abc.abstractmethod
    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        raise NotImplementedError

    # todo: state positional-only?
    def _strategy_begin(self, state, /, *, dt, parameters, vector_field):
        # todo: wrap those outputs into a _State type.
        state_strategy = self.strategy.begin(
            state.strategy,
            t=state.t,
            dt=dt,
            parameters=parameters,
            vector_field=vector_field,
        )
        return state_strategy

    def _strategy_complete(self, output_extra, state, /, *, cache_obs, output_scale):
        return self.strategy.complete(
            output_extra,
            state.strategy,
            output_scale=output_scale,
            cache_obs=cache_obs,
        )

    @abc.abstractmethod
    def extract_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_values_fn(self, state: _State, /) -> solution.Solution:
        raise NotImplementedError

    def solution_from_tcoeffs(
        self, taylor_coefficients, /, t, output_scale, num_data_points=1.0
    ):
        """Construct an initial `Solution` object.

        An (even if empty) solution object is needed to initialise the solver.
        Thus, this method is kind-of a helper function to make the rest of the
        initialisation code a bit simpler.
        """
        # todo: this should not call init(), but strategy.sol_from_tcoeffs()!
        posterior = self.strategy.solution_from_tcoeffs(
            taylor_coefficients, num_data_points=num_data_points
        )
        u = taylor_coefficients[0]

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
            num_data_points=self.strategy.num_data_points(posterior),
        )

    def init(self, sol, /) -> _State:
        error_estimate = self.strategy.init_error_estimate()
        strategy_state = self.strategy.init(sol.t, sol.posterior)
        return _State(
            u=sol.u,
            error_estimate=error_estimate,
            strategy=strategy_state,
            output_scale_prior=sol.output_scale,
            output_scale_calibrated=sol.output_scale,
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
        return jax.lax.switch(apply_branch, branches, t, s0, s1)

    def case_interpolate(self, t, s0: _State, s1: _State) -> InterpRes[_State]:
        acc_p, sol_p, prev_p = self.strategy.case_interpolate(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            # always interpolate with the prior output scale.
            #  This is important to make the MLE solver behave correctly.
            #  (Dynamic solvers overwrite the prior output scale at every step anyway).
            output_scale=s1.output_scale_prior,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return InterpRes(accepted=acc, solution=sol, previous=prev)

    def case_right_corner(self, t, s0: _State, s1: _State) -> InterpRes[_State]:
        acc_p, sol_p, prev_p = self.strategy.case_right_corner(
            t,
            s0=s0.strategy,
            s1=s1.strategy,
            output_scale=s1.output_scale_prior,
        )
        prev = self._interp_make_state(prev_p, reference=s0)
        sol = self._interp_make_state(sol_p, reference=s1)
        acc = self._interp_make_state(acc_p, reference=s1)
        return InterpRes(accepted=acc, solution=sol, previous=prev)

    def _interp_make_state(self, state_strategy, *, reference: _State) -> _State:
        error_estimate = self.strategy.init_error_estimate()
        u = self.strategy.extract_u(state=state_strategy)
        return _State(
            strategy=state_strategy,
            u=u,
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
        output_extra, (error, _, cache_obs) = self._strategy_begin(
            state, dt=dt, parameters=parameters, vector_field=vector_field
        )
        _, corrected = self._strategy_complete(
            output_extra,
            state,
            cache_obs=cache_obs,
            output_scale=state.output_scale_prior,
        )

        # Extract and return solution
        u = self.strategy.extract_u(state=corrected)
        return _State(
            u=u,
            error_estimate=dt * error,
            strategy=corrected,
            output_scale_prior=state.output_scale_prior,
            # Nothing happens in the field below:
            #  but we cannot use "None" if we want to reuse the init()
            #  method from abstract solvers (which populate this field).
            output_scale_calibrated=state.output_scale_prior,
        )

    def extract_fn(self, state: _State, /) -> solution.Solution:
        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            # _prior and _calibrated are identical.
            #  but we use _prior because we might remove the _calibrated
            #  value in the future.
            output_scale=state.output_scale_prior,
            num_data_points=self.strategy.num_data_points(state.strategy),
        )

    def extract_terminal_values_fn(self, state: _State, /) -> solution.Solution:
        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_prior,
            num_data_points=self.strategy.num_data_points(state.strategy),
        )


@jax.tree_util.register_pytree_node_class
class DynamicSolver(AbstractSolver):
    """Initial value problem solver with dynamic calibration of the output scale."""

    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        output_extra, (error, output_scale, cache_obs) = self._strategy_begin(
            state, dt=dt, parameters=parameters, vector_field=vector_field
        )
        _, corrected = self._strategy_complete(
            output_extra,
            state,
            cache_obs=cache_obs,
            output_scale=output_scale,
        )

        # Return solution
        u = self.strategy.extract_u(state=corrected)
        return _State(
            u=u,
            error_estimate=dt * error,
            strategy=corrected,
            output_scale_calibrated=output_scale,
            # current scale becomes the new prior scale!
            #  this is because dynamic solvers assume a piecewise-constant model
            output_scale_prior=output_scale,
        )

    def extract_fn(self, state: _State, /) -> solution.Solution:
        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=self.strategy.num_data_points(state.strategy),
        )

    def extract_terminal_values_fn(self, state: _State, /) -> solution.Solution:
        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)
        u = marginals.extract_qoi()
        return solution.Solution(
            t=t,
            u=u,  # new!
            marginals=marginals,  # new!
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=self.strategy.num_data_points(state.strategy),
        )


@jax.tree_util.register_pytree_node_class
class MLESolver(AbstractSolver):
    """Initial value problem solver with (quasi-)maximum-likelihood \
     calibration of the output-scale."""

    def step_fn(self, *, state: _State, vector_field, dt, parameters) -> _State:
        output_extra, (error, _, cache_obs) = self._strategy_begin(
            state, dt=dt, parameters=parameters, vector_field=vector_field
        )
        observed, corrected = self._strategy_complete(
            output_extra,
            state,
            cache_obs=cache_obs,
            output_scale=state.output_scale_prior,
        )

        # Calibrate
        output_scale = state.output_scale_calibrated
        n = self.strategy.num_data_points(state.strategy)
        new_output_scale = self._update_output_scale(
            diffsqrtm=output_scale, n=n, obs=observed
        )

        # Extract and return solution
        u = self.strategy.extract_u(state=corrected)
        return _State(
            u=u,
            error_estimate=dt * error,
            strategy=corrected,
            output_scale_prior=state.output_scale_prior,
            output_scale_calibrated=new_output_scale,
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
        sum_updated = _sqrt_util.sqrt_sum_square(jnp.sqrt(n) * diffsqrtm, x)
        return sum_updated / jnp.sqrt(n + 1)

    def extract_fn(self, state: _State, /) -> solution.Solution:
        # 'state' is batched. Thus, output scale is an array instead of a scalar.

        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals(posterior)

        # promote calibrated scale to the correct batch-shape
        s = state.output_scale_calibrated[-1] * jnp.ones_like(state.output_scale_prior)
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=t,
            u=state.u,
            marginals=marginals,
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=self.strategy.num_data_points(state.strategy),
        )

    def extract_terminal_values_fn(self, state: _State, /) -> solution.Solution:
        # 'state' is not batched. Thus, output scale is a scalar.

        t, posterior = self.strategy.extract(state.strategy)
        marginals = self.strategy.extract_marginals_terminal_values(posterior)

        s = state.output_scale_calibrated
        state = self._rescale_covs(state, output_scale=s, marginals_unscaled=marginals)
        return solution.Solution(
            t=t,
            u=state.u,
            marginals=marginals,
            posterior=posterior,
            output_scale=state.output_scale_calibrated,
            num_data_points=self.strategy.num_data_points(state.strategy),
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
        #  posterior = scale_cov(state.strategy)
        #  Which would avoid having to do this over and over again
        #  in intermediate objects
        #  (Conditionals, Posteriors, StateSpaceVars, ...)

        marginals = marginals_unscaled.scale_covariance(output_scale)
        state_strategy = state.strategy.scale_covariance(output_scale)
        u = marginals.extract_qoi()
        return _State(
            u=u,
            strategy=state_strategy,
            output_scale_calibrated=output_scale,
            output_scale_prior=None,  # irrelevant, will be removed in next step
            error_estimate=None,  # irrelevant, will be removed in next step.
        )
