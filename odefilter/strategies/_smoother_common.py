"""."""


import abc
from dataclasses import dataclass

import jax
import jax.tree_util

from odefilter.strategies import _interface, markov


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicSmootherCommon(_interface.Strategy):
    """Common functionality for smoother-style algorithms."""

    @abc.abstractmethod
    def _case_interpolate(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_right_corner(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def init_fn(self, taylor_coefficients, t0):
        raise NotImplementedError

    def extract_fn(self, *, state):  # noqa: D102
        # todo: are we looping correctly?
        #  what does the backward transition at time t say?
        #  How to get from t to the previous t, right?

        # no jax.lax.cond here, because we condition on the _shape_ of the array
        # which is available at compilation time already.
        do_backward_pass = state.t.ndim == 1
        if do_backward_pass:
            return self._smooth(state)

        return markov.Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals_filtered=state.marginals_filtered,
            marginals=state.marginals_filtered,  # we are at the terminal state only
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )

    def _smooth(self, state):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.marginals_filtered)
        marginals = self.implementation.marginalise_backwards(
            init=init,
            linop=state.backward_model.transition,
            noise=state.backward_model.noise,
        )
        sol = self.implementation.extract_sol(rv=marginals)
        return markov.Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=sol,
            marginals_filtered=state.marginals_filtered,
            marginals=marginals,
            diffusion_sqrtm=state.diffusion_sqrtm,
            backward_model=state.backward_model,
        )

    def _duplicate_with_unit_backward_model(self, s0, t):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=s0.backward_model.noise
        )
        bw_model = markov.BackwardModel(transition=bw_transition0, noise=bw_noise0)
        state1 = markov.Posterior(
            t=t,
            t_previous=t,  # identity transition: this is what it does...
            u=s0.u,
            marginals_filtered=s0.marginals_filtered,
            marginals=s0.marginals,
            diffusion_sqrtm=s0.diffusion_sqrtm,
            backward_model=bw_model,
        )
        return state1

    def _interpolate_from_to_fn(self, rv, diffusion_sqrtm, t, t0):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            rv.mean, p=p, p_inv=p_inv
        )

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=rv.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = markov.BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model

    # Not implemented yet:

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
