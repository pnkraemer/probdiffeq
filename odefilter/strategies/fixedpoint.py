"""Inference via fixed-point smoothing."""


import jax
import jax.tree_util

from odefilter.strategies import _common


@jax.tree_util.register_pytree_node_class
class FixedPointSmootherStrategy(_common.SmootherStrategyCommon):
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
        bw_increment = _common.BackwardModel(transition=bw_op, noise=bw_noise)

        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=posterior_previous.backward_model,
        )
        backward_model = _common.BackwardModel(transition=gain, noise=noise)

        return _common.MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, s0, s1, t):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        backward_model1 = s1.posterior.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.posterior.backward_model, bw_state=backward_model1
        )
        backward_model1 = _common.BackwardModel(transition=g0, noise=noise0)
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        solution = _common.Solution(
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
        backward_model0 = _common.BackwardModel(transition=g0, noise=noise0)
        posterior0 = _common.MarkovSequence(
            init=extrapolated0, backward_model=backward_model0
        )
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = _common.Solution(
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
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        accepted = _common.Solution(
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


@jax.tree_util.register_pytree_node_class
class DynamicFixedPointSmoother(_common.DynamicSolver):
    """Filter implementation (time-constant output-scale)."""

    def __init__(self, *, implementation):
        strategy = FixedPointSmootherStrategy(implementation=implementation)
        super().__init__(strategy=strategy)

    def tree_flatten(self):
        children = (self.strategy.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)
