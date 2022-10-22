"""Inference via smoothing."""

import abc
from typing import Any, Generic, TypeVar

import jax
import jax.tree_util

from odefilter.strategies import _strategy, solvers

# todo: nothing in here should operate on "Solution"-types!


T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
class BackwardModel(Generic[T]):
    """Backward model for backward-Gauss--Markov process representations."""

    def __init__(self, *, transition: Any, noise: T):
        self.transition = transition
        self.noise = noise

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[T]):
    """Markov sequence."""

    def __init__(self, *, init: T, backward_model: BackwardModel[T]):
        self.init = init
        self.backward_model = backward_model

    def tree_flatten(self):
        children = (self.init, self.backward_model)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model = children
        return cls(init=init, backward_model=backward_model)


class _SmootherCommon(_strategy.Strategy):
    """Common functionality for smoothers."""

    # Inherited abstract methods

    @abc.abstractmethod
    def case_interpolate(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, m_ext, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        raise NotImplementedError

    def init_posterior(self, *, corrected):
        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        posterior = MarkovSequence(init=corrected, backward_model=backward_model)
        return posterior

    def extrapolate_mean(self, *, posterior, p_inv, p):
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            posterior.init.mean, p=p, p_inv=p_inv
        )
        return m_ext, (m_ext_p, m0_p)

    def final_correction(self, *, extrapolated, linear_fn, m_obs):
        a, (corrected, b) = self.implementation.final_correction(
            extrapolated=extrapolated.init, linear_fn=linear_fn, m_obs=m_obs
        )
        corrected_seq = MarkovSequence(
            init=corrected, backward_model=extrapolated.backward_model
        )
        return a, (corrected_seq, b)

    def extract_sol_terminal_value(self, *, posterior):
        return self.implementation.extract_sol(rv=posterior.init)

    def extract_sol_from_marginals(self, *, marginals):
        return self.implementation.extract_sol(rv=marginals)

    def marginals_terminal_value(self, *, posterior):
        return posterior.init

    def marginals(self, *, posterior):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        marginals = self.implementation.marginalise_backwards(
            init=init,
            linop=posterior.backward_model.transition,
            noise=posterior.backward_model.noise,
        )
        return marginals

    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=marginals, scale_sqrtm=output_scale_sqrtm
        )

    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        init = self.implementation.scale_covariance(
            rv=posterior.init, scale_sqrtm=output_scale_sqrtm
        )
        noise = self.implementation.scale_covariance(
            rv=posterior.backward_model.noise, scale_sqrtm=output_scale_sqrtm
        )

        bw_model = BackwardModel(
            transition=posterior.backward_model.transition, noise=noise
        )
        return MarkovSequence(init=init, backward_model=bw_model)

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            rv.mean, p=p, p_inv=p_inv
        )

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            m_ext_p=m_ext_p,
            m0_p=m0_p,
            l0=rv.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model  # should this return a MarkovSequence?

    def _duplicate_with_unit_backward_model(self, *, state, t):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=state.posterior.backward_model.noise
        )
        bw_model = BackwardModel(transition=bw_transition0, noise=bw_noise0)

        posterior = MarkovSequence(init=state.posterior.init, backward_model=bw_model)
        state1 = solvers.Solution(
            t=t,
            t_previous=t,  # identity transition: this is what it does...
            u=state.u,
            posterior=posterior,
            marginals=state.marginals,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )
        return state1


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self, m_ext, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        m_ext_p, m0_p = cache
        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=posterior_previous.init.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, s0, s1, t):  # s1.t == t
        accepted = self._duplicate_with_unit_backward_model(state=s1, t=t)
        previous = solvers.Solution(
            t=t,
            t_previous=s0.t,
            u=s1.u,
            posterior=s1.posterior,
            marginals=None,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        solution = previous

        return accepted, solution, previous

    def case_interpolate(self, *, s0, s1, t):
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        rv0, diffsqrtm = s0.posterior.init, s1.output_scale_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=rv0, output_scale_sqrtm=diffsqrtm, t=t, t0=s0.t
        )
        posterior0 = MarkovSequence(init=extrapolated0, backward_model=backward_model0)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=diffsqrtm, t=s1.t, t0=t
        )
        posterior1 = MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = solvers.Solution(
            t=t,
            t_previous=s0.t,
            u=sol,
            posterior=posterior0,
            marginals=None,
            output_scale_sqrtm=diffsqrtm,
            num_data_points=s1.num_data_points,
        )

        # This is what we will interpolate from next.
        # The backward model needs no resetting, because the smoother
        # does not condense.
        previous = solution

        accepted = solvers.Solution(
            t=s1.t,
            t_previous=t,
            u=s1.u,
            posterior=posterior1,
            marginals=s1.marginals,
            output_scale_sqrtm=diffsqrtm,
            num_data_points=s1.num_data_points,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, state_previous, t, state):
        acc, _sol, _prev = self.case_interpolate(t=t, s1=state, s0=state_previous)
        marginals = self.implementation.marginalise_model(
            init=acc.marginals,
            linop=acc.posterior.backward_model.transition,
            noise=acc.posterior.backward_model.noise,
        )
        u = self.extract_sol_from_marginals(marginals=marginals)
        return u, marginals


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother."""

    def complete_extrapolation(
        self, m_ext, cache, *, posterior_previous, output_scale_sqrtm, p, p_inv
    ):
        m_ext_p, m0_p = cache
        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=posterior_previous.init.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        bw_increment = BackwardModel(transition=bw_op, noise=bw_noise)

        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=posterior_previous.backward_model,
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, s0, s1, t):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        backward_model1 = s1.posterior.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.posterior.backward_model, bw_state=backward_model1
        )
        backward_model1 = BackwardModel(transition=g0, noise=noise0)
        posterior1 = MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        solution = solvers.Solution(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=s1.u,
            posterior=posterior1,
            marginals=None,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )

        accepted = self._duplicate_with_unit_backward_model(state=solution, t=t)
        previous = accepted

        return accepted, solution, previous

    def case_interpolate(self, *, s0, s1, t):  # noqa: D102
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # Use the s1.output-scale as a output-scale over the interval.
        # Filtering/smoothing solutions are right-including intervals.
        output_scale_sqrtm = s1.output_scale_sqrtm

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.posterior.init,
            output_scale_sqrtm=output_scale_sqrtm,
            t=t,
            t0=s0.t,
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.posterior.backward_model, bw_state=bw0
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)
        posterior0 = MarkovSequence(init=extrapolated0, backward_model=backward_model0)
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = solvers.Solution(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=sol,
            posterior=posterior0,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )

        # This is what we interpolate from next.
        previous = self._duplicate_with_unit_backward_model(state=solution, t=t)

        # From t to s1.t
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=output_scale_sqrtm, t=s1.t, t0=t
        )
        posterior1 = MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        accepted = solvers.Solution(
            t=s1.t,
            t_previous=t,  # new model! No condensing...
            u=s1.u,
            posterior=posterior1,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
