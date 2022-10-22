"""Inference via smoothing."""


import jax
import jax.tree_util

from odefilter.strategies import _common


@jax.tree_util.register_pytree_node_class
class SmootherStrategy(_common.SmootherStrategyCommon):
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
        backward_model = _common.BackwardModel(transition=bw_op, noise=bw_noise)
        return _common.MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, s0, s1, t):  # s1.t == t
        accepted = self._duplicate_with_unit_backward_model(state=s1, t=t)
        previous = _common.Solution(
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
        posterior0 = _common.MarkovSequence(
            init=extrapolated0, backward_model=backward_model0
        )

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=diffsqrtm, t=s1.t, t0=t
        )
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )

        # This is the new solution object at t.
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = _common.Solution(
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

        accepted = _common.Solution(
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
class DynamicSmoother(_common.DynamicSolver):
    """Filter implementation (time-constant output-scale)."""

    def __init__(self, *, implementation):
        strategy = SmootherStrategy(implementation=implementation)
        super().__init__(strategy=strategy)

    def tree_flatten(self):
        children = (self.strategy.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)
