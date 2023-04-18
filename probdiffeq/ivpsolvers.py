"""Calibrated IVP solvers."""

import abc
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util, solution
from probdiffeq._collections import InterpRes  # simplify type signatures


class _State(NamedTuple):
    """Solver state."""

    strategy: Any

    error_estimate: Any
    output_scale_calibrated: Any
    output_scale_prior: Any

    num_steps: Any

    @property
    def t(self):
        return self.strategy.t

    @property
    def u(self):
        return self.strategy.u


@jax.tree_util.register_pytree_node_class
class Solver(abc.ABC):
    """Interface for initial value problem solvers."""

    def __init__(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return f"{self.__class__.__name__}({self.strategy})"

    #  methods

    @abc.abstractmethod
    def step(self, *, state: _State, vector_field, dt, parameters) -> _State:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: _State, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_at_terminal_values(self, state: _State, /):
        raise NotImplementedError

    def solution_from_tcoeffs(
        self, taylor_coefficients, /, t, output_scale, num_steps=1.0
    ):
        """Construct an initial `Solution` object.

        An (even if empty) solution object is needed to initialise the solver.
        Thus, this method is kind-of a helper function to make the rest of the
        initialisation code a bit simpler.
        """
        u, marginals, posterior = self.strategy.solution_from_tcoeffs(
            taylor_coefficients
        )
        # todo: remove!?
        output_scale = self.strategy.promote_output_scale(output_scale)
        return solution.Solution(
            t=t,
            posterior=posterior,
            marginals=marginals,
            output_scale=output_scale,
            u=u,
            num_steps=num_steps,
        )

    def init(self, t, posterior, /, output_scale, num_steps) -> _State:
        error_estimate = self.strategy.init_error_estimate()
        strategy_state = self.strategy.init(t, None, None, posterior)
        return _State(
            error_estimate=error_estimate,
            strategy=strategy_state,
            output_scale_prior=output_scale,
            output_scale_calibrated=output_scale,
            num_steps=num_steps,
        )

    def interpolate(self, *, s0: _State, s1: _State, t):
        # Cases to switch between
        branches = [self._case_right_corner, self._case_interpolate]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10  # todo: magic constant?
        is_in_between = jnp.logical_not(is_right_corner)
        index_as_array = jnp.asarray([is_right_corner, is_in_between])

        # Select branch and return result
        apply_branch_as_array, *_ = jnp.where(index_as_array, size=1)
        apply_branch = jnp.reshape(apply_branch_as_array, ())
        return jax.lax.switch(apply_branch, branches, t, s0, s1)

    def _case_interpolate(self, t, s0: _State, s1: _State) -> InterpRes[_State]:
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

    def _case_right_corner(self, t, s0: _State, s1: _State) -> InterpRes[_State]:
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
        return _State(
            strategy=state_strategy,
            error_estimate=error_estimate,
            output_scale_prior=reference.output_scale_prior,
            output_scale_calibrated=reference.output_scale_calibrated,
            num_steps=reference.num_steps,
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
class CalibrationFreeSolver(Solver):
    """Initial value problem solver.

    No automatic output-scale calibration.
    """

    def step(self, *, state: _State, vector_field, dt, parameters) -> _State:
        state_strategy = self.strategy.begin(
            state.strategy,
            dt=dt,
            parameters=parameters,
            vector_field=vector_field,
        )
        (error, _, cache_obs) = state_strategy.corr

        state_strategy = self.strategy.complete(
            state_strategy,
            parameters=parameters,
            vector_field=vector_field,
            output_scale=state.output_scale_prior,
        )

        # Extract and return solution
        return _State(
            error_estimate=dt * error,
            strategy=state_strategy,
            output_scale_prior=state.output_scale_prior,
            # Nothing happens in the field below:
            #  but we cannot use "None" if we want to reuse the init()
            #  method from abstract solvers (which populate this field).
            output_scale_calibrated=state.output_scale_prior,
            num_steps=state.num_steps + 1,
        )

    def extract(self, state: _State, /):
        t, posterior = self.strategy.extract(state.strategy)
        return t, posterior, state.output_scale_prior, state.num_steps

    def extract_at_terminal_values(self, state: _State, /):
        t, posterior = self.strategy.extract_at_terminal_values(state.strategy)
        return t, posterior, state.output_scale_prior, state.num_steps


@jax.tree_util.register_pytree_node_class
class DynamicSolver(Solver):
    """Initial value problem solver with dynamic calibration of the output scale."""

    def step(self, *, state: _State, vector_field, dt, parameters) -> _State:
        state_strategy = self.strategy.begin(
            state.strategy,
            dt=dt,
            parameters=parameters,
            vector_field=vector_field,
        )
        (error, output_scale, _) = state_strategy.corr  # clean this up next?

        state_strategy = self.strategy.complete(
            state_strategy,
            parameters=parameters,
            vector_field=vector_field,
            output_scale=output_scale,
        )

        # Return solution
        return _State(
            error_estimate=dt * error,
            strategy=state_strategy,
            output_scale_calibrated=output_scale,
            # current scale becomes the new prior scale!
            #  this is because dynamic solvers assume a piecewise-constant model
            output_scale_prior=output_scale,
            num_steps=state.num_steps + 1,
        )

    def extract(self, state: _State, /):
        t, posterior = self.strategy.extract(state.strategy)
        return t, posterior, state.output_scale_calibrated, state.num_steps

    def extract_at_terminal_values(self, state: _State, /):
        t, posterior = self.strategy.extract_at_terminal_values(state.strategy)
        return t, posterior, state.output_scale_calibrated, state.num_steps


@jax.tree_util.register_pytree_node_class
class MLESolver(Solver):
    """Initial value problem solver with (quasi-)maximum-likelihood \
     calibration of the output-scale."""

    def step(self, *, state: _State, vector_field, dt, parameters) -> _State:
        state_strategy = self.strategy.begin(
            state.strategy,
            dt=dt,
            parameters=parameters,
            vector_field=vector_field,
        )
        (error, output_scale, _) = state_strategy.corr  # clean this up next?

        state_strategy = self.strategy.complete(
            state_strategy,
            output_scale=state.output_scale_prior,
            parameters=parameters,
            vector_field=vector_field,
        )
        observed = state_strategy.corr  # clean this up next?

        # Calibrate
        output_scale = state.output_scale_calibrated
        n = state.num_steps
        new_output_scale = self._update_output_scale(
            diffsqrtm=output_scale, n=n, obs=observed
        )
        return _State(
            error_estimate=dt * error,
            strategy=state_strategy,
            output_scale_prior=state.output_scale_prior,
            output_scale_calibrated=new_output_scale,
            num_steps=state.num_steps + 1,
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

    def extract(self, state: _State, /):
        # 'state' is batched. Thus, output scale is an array instead of a scalar.

        # Important: Rescale before extracting! Otherwise backward samples are wrong.

        # promote calibrated scale to the correct batch-shape
        s = state.output_scale_calibrated[-1] * jnp.ones_like(state.output_scale_prior)
        state = self._rescale_covs(state, output_scale=s)

        t, posterior = self.strategy.extract(state.strategy)
        return t, posterior, state.output_scale_calibrated, state.num_steps

    def extract_at_terminal_values(self, state: _State, /):
        # 'state' is not batched. Thus, output scale is a scalar.
        # Important: Rescale before extracting! Otherwise backward samples are wrong.
        s = state.output_scale_calibrated
        state = self._rescale_covs(state, output_scale=s)

        t, posterior = self.strategy.extract_at_terminal_values(state.strategy)
        # t, u, marginals, posterior = _sol
        return t, posterior, state.output_scale_calibrated, state.num_steps

    def _rescale_covs(self, state, /, *, output_scale):
        # todo: these calls to *.scale_covariance are a bit cumbersome,
        #  because we need to add this
        #  method to all sorts of classes.
        #  Instead, we could move the collections.Normal
        #  to the top-level and implement this via tree_map:
        #  def scale_cov(tree):
        #      def is_leaf(x):
        #          return isinstance(x, Normal)
        #      def fn(x: Normal):
        #          return x.scale_covariance(output_scale)
        #      return tree_map(fn, tree, is_leaf=is_leaf)
        #  marginals = scale_cov(marginals_unscaled)
        #  posterior = scale_cov(state.strategy)
        #  Which would avoid having to do this over and over again
        #  in intermediate objects
        #  (Conditionals, Posteriors, StateSpaceVars, ...)

        state_strategy = state.strategy.scale_covariance(output_scale)
        state_rescaled = _State(
            strategy=state_strategy,
            output_scale_calibrated=output_scale,
            output_scale_prior=None,  # irrelevant, will be removed in next step
            error_estimate=None,  # irrelevant, will be removed in next step.
            num_steps=state.num_steps,
        )
        return state_rescaled
