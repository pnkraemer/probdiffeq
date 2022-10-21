"""Inference interface."""

import abc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from odefilter.strategies import _markov


@dataclass(frozen=True)
class Strategy(abc.ABC):
    """Inference strategy interface."""

    implementation: Any

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    @abc.abstractmethod
    def init_fn(self, *, taylor_coefficients, t0):  # -> state
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):  # -> solution
        # Don't jit the extract fun unless it is performance-critical!
        # Why? Because it would recompile every time the ODE parameters
        # change in solve(), provided the change is sufficient to change
        # the number of time-steps taken.
        # In the fully-jit-able functions, it is compiled automatically anyway.
        raise NotImplementedError

    @abc.abstractmethod
    def extract_terminal_value_fn(self, *, state):  # -> solution
        raise NotImplementedError

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102

        # Cases to switch between
        branches = [
            lambda s0, s1, t: self._case_right_corner(s0=s0, s1=s1, t=t),
            lambda s0, s1, t: self._case_interpolate(s0=s0, s1=s1, t=t),
        ]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10
        is_in_between = jnp.logical_not(is_right_corner)

        index_as_array, *_ = jnp.where(
            jnp.asarray([is_right_corner, is_in_between]), size=1
        )
        index = jnp.reshape(index_as_array, ())
        return jax.lax.switch(index, branches, s0, s1, t)

    def offgrid_marginals_searchsorted(self, *, ts, solution):
        """Dense output for a whole grid via jax.numpy.searchsorted.

        !!! warning
            The elements in ts and the elements in the solution grid must be disjoint.
            Otherwise, anything can happen and the solution will be incorrect.
            We do not check for this case! (Because we want to jit!)

        !!! warning
            The elements in ts must be strictly in (t0, t1).
            Again there is no check and anything can happen if you don't follow
            this rule.
        """
        # todo: support "method" argument.

        # side="left" and side="right" are equivalent
        # because we _assume_ that the point sets are disjoint.
        indices = jnp.searchsorted(solution.t, ts)

        # Solution slicing to the rescue
        solution_left = solution[indices - 1]
        solution_right = solution[indices]

        # Vmap to the rescue :) It does not like kw-only arguments, though.
        def offgrid_no_kw(sprev, t, s):
            return self.offgrid_marginals(t=t, state=s, state_previous=sprev)

        marginals_vmap = jax.vmap(offgrid_no_kw)
        return marginals_vmap(solution_left, ts, solution_right)

    @abc.abstractmethod
    def _case_right_corner(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_interpolate(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicSmootherCommon(Strategy):
    """Common functionality for smoother-style algorithms."""

    @abc.abstractmethod
    def _case_interpolate(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_right_corner(self, *, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def init_fn(self, *, taylor_coefficients, t0):
        raise NotImplementedError

    def extract_fn(self, *, state):  # noqa: D102
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.marginals_filtered)
        marginals = self.implementation.marginalise_backwards(
            init=init,
            linop=state.backward_model.transition,
            noise=state.backward_model.noise,
        )
        sol = self.implementation.extract_sol(rv=marginals)
        return _markov.Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=sol,
            marginals_filtered=state.marginals_filtered,
            marginals=marginals,
            output_scale_sqrtm=state.output_scale_sqrtm,
            backward_model=state.backward_model,
        )

    def extract_terminal_value_fn(self, *, state):  # noqa: D102
        return _markov.Posterior(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            marginals_filtered=state.marginals_filtered,
            marginals=state.marginals_filtered,  # we are at the terminal state only
            output_scale_sqrtm=state.output_scale_sqrtm,
            backward_model=state.backward_model,
        )

    def _duplicate_with_unit_backward_model(self, *, state, t):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=state.backward_model.noise
        )
        bw_model = _markov.BackwardModel(transition=bw_transition0, noise=bw_noise0)
        state1 = _markov.Posterior(
            t=t,
            t_previous=t,  # identity transition: this is what it does...
            u=state.u,
            marginals_filtered=state.marginals_filtered,
            marginals=state.marginals,
            output_scale_sqrtm=state.output_scale_sqrtm,
            backward_model=bw_model,
        )
        return state1

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
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
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = _markov.BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model

    # Not implemented yet:

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
