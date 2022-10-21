"""Inference interface."""

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BackwardModel(Generic[T]):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: T

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MarkovSequence(Generic[T]):
    """Markov sequence."""

    init: T
    backward_model: BackwardModel[T]

    def tree_flatten(self):
        children = (self.init, self.backward_model)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model = children
        return cls(init=init, backward_model=backward_model)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Solution(Generic[T]):
    """Inferred solutions."""

    t: float
    t_previous: float

    u: Any
    marginals: T

    output_scale_sqrtm: float

    # todo: either marginals or posterior are plenty?
    #  I think the posterior should be the only one, but
    #  then, interpolate() and offgrid_marginals() should only return marginals.
    #  Otherwise, this doesn't really work.
    posterior: MarkovSequence[T]

    num_data_points: float  # todo: make int

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
            self.u,
            self.marginals,
            self.posterior,
            self.output_scale_sqrtm,
            self.num_data_points,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        t, t_previous, u, marginals, posterior, output_scale_sqrtm, n = children
        return cls(
            t=t,
            t_previous=t_previous,
            u=u,
            marginals=marginals,
            posterior=posterior,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=n,
        )

    def __len__(self):
        """Length of a solution object.

        Depends on the length of the underlying :attr:`t` attribute.
        """
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access the `i`-th sub-solution."""
        if jnp.ndim(self.t) < 1:
            raise ValueError(f"Solution object not batched :(, {jnp.ndim(self.t)}")
        if isinstance(item, tuple) and len(item) > jnp.ndim(self.t):
            # s[2, 3] forbidden
            raise ValueError(f"Inapplicable shape: {item, jnp.shape(self.t)}")
        return Solution(
            t=self.t[item],
            t_previous=self.t_previous[item],
            u=self.u[item],
            output_scale_sqrtm=self.output_scale_sqrtm[item],
            # todo: make iterable?
            marginals=jax.tree_util.tree_map(lambda x: x[item], self.marginals),
            # todo: make iterable?
            posterior=jax.tree_util.tree_map(lambda x: x[item], self.posterior),
            num_data_points=self.num_data_points[item],
        )

    def __iter__(self):
        """Iterate through the filtering solution."""
        for i in range(self.t.shape[0]):
            yield self[i]


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
            lambda s0_, s1_, t_: self._case_right_corner(s0=s0_, s1=s1_, t=t_),
            lambda s0_, s1_, t_: self._case_interpolate(s0=s0_, s1=s1_, t=t_),
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
        # todo: return _only_ the marginals

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
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], state.posterior.init)
        marginals = self.implementation.marginalise_backwards(
            init=init,
            linop=state.posterior.backward_model.transition,
            noise=state.posterior.backward_model.noise,
        )
        sol = self.implementation.extract_sol(rv=marginals)
        return Solution(
            t=state.t,
            t_previous=state.t_previous,
            u=sol,
            posterior=state.posterior,
            marginals=marginals,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def extract_terminal_value_fn(self, *, state):  # noqa: D102
        return Solution(
            t=state.t,
            t_previous=state.t_previous,
            u=state.u,
            posterior=state.posterior,
            marginals=state.posterior.init,  # we are at the terminal state only
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
        )

    def _duplicate_with_unit_backward_model(self, *, state, t):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=state.posterior.backward_model.noise
        )
        bw_model = BackwardModel(transition=bw_transition0, noise=bw_noise0)
        posterior = MarkovSequence(init=state.posterior.init, backward_model=bw_model)
        state1 = Solution(
            t=t,
            t_previous=t,  # identity transition: this is what it does...
            u=state.u,
            posterior=posterior,
            marginals=state.marginals,
            output_scale_sqrtm=state.output_scale_sqrtm,
            num_data_points=state.num_data_points,
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
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model  # should this return a MarkovSequence?

    # Not implemented yet:

    @abc.abstractmethod
    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
