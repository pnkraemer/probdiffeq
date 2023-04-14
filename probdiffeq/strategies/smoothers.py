"""''Global'' estimation: smoothing."""

import abc
import functools
from typing import Any, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _control_flow
from probdiffeq._collections import InterpRes
from probdiffeq.strategies import _strategy

S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""

# todo: markov sequences should not necessarily be backwards


@jax.tree_util.register_pytree_node_class
class MarkovSequence(_strategy.Posterior[S]):
    """Markov sequence. A discretised Markov process."""

    def __init__(self, *, init: S, backward_model, num_data_points):
        self.init = init
        self.backward_model = backward_model
        self.num_data_points = num_data_points

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(init={self.init}, backward_model={self.backward_model})"

    def tree_flatten(self):
        children = (self.init, self.backward_model, self.num_data_points)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model, n = children
        return cls(init=init, backward_model=backward_model, num_data_points=n)

    def sample(self, key, *, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        base_samples = jax.random.normal(key=key, shape=shape + self.sample_shape)
        return self._transform_unit_sample(base_samples)

    def _transform_unit_sample(self, base_sample, /):
        if base_sample.shape == self.sample_shape:
            return self._transform_one_unit_sample(base_sample)

        transform_vmap = jax.vmap(self._transform_unit_sample, in_axes=0)
        return transform_vmap(base_sample)

    def _transform_one_unit_sample(self, base_sample, /):
        def body_fun(carry, conditionals_and_base_samples):
            _, samp_prev = carry
            conditional, base = conditionals_and_base_samples

            cond = conditional(samp_prev)
            samp = cond.hidden_state.transform_unit_sample(base)
            qoi = cond.extract_qoi_from_sample(samp)

            return (qoi, samp), (qoi, samp)

        # Compute a sample at the terminal value
        init = jax.tree_util.tree_map(lambda s: s[-1, ...], self.init)
        init_sample = init.hidden_state.transform_unit_sample(base_sample[-1])
        init_qoi = init.extract_qoi_from_sample(init_sample)
        init_val = (init_qoi, init_sample)

        # Remove the initial backward-model
        conds = jax.tree_util.tree_map(lambda s: s[1:, ...], self.backward_model)

        # Loop over backward models and the remaining base samples
        xs = (conds, base_sample[:-1])
        _, (qois, samples) = jax.lax.scan(
            f=body_fun, init=init_val, xs=xs, reverse=True
        )
        qois_full = jnp.concatenate((qois, init_qoi[None, ...]))
        samples_full = jnp.concatenate((samples, init_sample[None, ...]))
        return qois_full, samples_full

    def marginalise_backwards(self):
        def body_fun(rv, conditional):
            out = conditional.marginalise(rv)
            return out, out

        # Initial condition does not matter
        conds = jax.tree_util.tree_map(lambda x: x[1:, ...], self.backward_model)

        # Scan and return
        reverse_scan = functools.partial(_control_flow.scan_with_init, reverse=True)
        _, rvs = reverse_scan(f=body_fun, init=self.init, xs=conds)
        return rvs

    @property
    def sample_shape(self):
        return self.backward_model.noise.sample_shape


_SolType = Tuple[float, jax.Array, jax.Array, MarkovSequence]


class _SmState(NamedTuple):
    t: Any
    u: Any
    ssv: Any

    # todo: move to StateSpaceVar
    #  (only the SLR knows that it needs `d` data points per step...)
    num_data_points: Any

    @property
    def error_estimate(self):
        return self.ssv.error_estimate

    def scale_covariance(self, output_scale):
        ssv = self.ssv.scale_covariance(output_scale)
        return _SmState(
            t=self.t,
            u=self.u,
            ssv=ssv,
            num_data_points=self.num_data_points,
        )


class _SmootherCommon(_strategy.Strategy):
    # Inherited abstract methods

    @abc.abstractmethod
    def case_interpolate(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior: MarkovSequence,
        posterior_previous: MarkovSequence,
        t0,
        t1,
        output_scale,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def _complete_extrapolation(
        self,
        output_extra: _SmState,
        /,
        *,
        output_scale,
        state_previous: _SmState,
    ):
        raise NotImplementedError

    def init(self, t, u, marginals, posterior, /) -> _SmState:
        ssv = self.extrapolation.init_with_reversal(
            posterior.init, posterior.backward_model
        )
        ssv = self.correction.init(ssv)
        return _SmState(
            t=t,
            u=u,
            ssv=ssv,
            num_data_points=posterior.num_data_points,
        )

    def solution_from_tcoeffs(self, taylor_coefficients, /, *, num_data_points):
        rv, cond = self.extrapolation.solution_from_tcoeffs_with_reversal(
            taylor_coefficients
        )
        #
        # init_bw_model = self.extrapolation.init_conditional
        # bw_model = init_bw_model(ssv_proto=corrected)
        sol = MarkovSequence(
            init=rv, backward_model=cond, num_data_points=num_data_points
        )
        marginals = rv
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, state: _SmState, /) -> _SolType:
        init, bw_model = self.extrapolation.extract_with_reversal(state.ssv)
        markov_seq = MarkovSequence(
            init=init,
            backward_model=bw_model,
            num_data_points=state.num_data_points,
        )
        marginals = self._extract_marginals(markov_seq)
        u = marginals.extract_qoi()
        return state.t, u, marginals, markov_seq

    def extract_at_terminal_values(self, state: _SmState, /) -> _SolType:
        init, bw_model = self.extrapolation.extract_with_reversal(state.ssv)
        markov_seq = MarkovSequence(
            init=init,
            backward_model=bw_model,
            num_data_points=state.num_data_points,
        )
        marginals = state.ssv.hidden_state
        u = marginals.extract_qoi()
        return state.t, u, marginals, markov_seq

    def _begin_extrapolation(self, posterior: _SmState, /, *, dt) -> _SmState:
        ssv = self.extrapolation.begin(posterior.ssv, dt=dt)
        return _SmState(
            t=posterior.t + dt,
            u=None,
            ssv=ssv,
            num_data_points=posterior.num_data_points,
        )

    def _begin_correction(
        self, x: _SmState, /, *, vector_field, p
    ) -> Tuple[jax.Array, float, Any]:
        ssv = self.correction.begin(x.ssv, vector_field, x.t, p)
        return _SmState(
            t=x.t,
            u=x.u,
            ssv=ssv,
            num_data_points=x.num_data_points,
        )

    def _complete_correction(self, x: _SmState, /, *, vector_field, p):
        ssv = self.correction.complete(x.ssv, vector_field, x.t, p)
        return _SmState(
            t=x.t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            num_data_points=x.num_data_points + 1,  # todo: move this info to SSV
        )

    def _extract_marginals(self, posterior: MarkovSequence, /):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        markov = MarkovSequence(
            init=init,
            backward_model=posterior.backward_model,
            num_data_points=posterior.num_data_points,
        )
        return markov.marginalise_backwards()

    def num_data_points(self, state, /):
        return state.num_data_points

    def observation(self, state, /):
        return state.ssv.observed_state

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale, t, t0):
        # todo: act on state instead of rv+t0
        dt = t - t0
        output_extra = self.extrapolation.begin(rv, dt=dt)
        return self.extrapolation.complete_with_reversal(
            output_extra,
            state_previous=rv,
            output_scale=output_scale,
        )

    # # todo: should this be a classmethod of MarkovSequence?
    # def _duplicate_with_unit_backward_model(self, posterior: _SmState, /) -> _SmState:
    #     init_conditional_fn = self.extrapolation.init_conditional
    #     bw_model = init_conditional_fn(ssv_proto=posterior.corrected)
    #     return _SmState(
    #         t=posterior.t,
    #         u=posterior.u,
    #         extrapolated=posterior.extrapolated,
    #         corrected=posterior.corrected,
    #         backward_model=bw_model,
    #         num_data_points=posterior.num_data_points,
    #     )


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def _complete_extrapolation(
        self,
        x: _SmState,
        /,
        *,
        output_scale,
        state_previous: _SmState,
    ) -> _SmState:
        ssv = self.extrapolation.complete_with_reversal(
            x.ssv,
            state_previous=state_previous.ssv,
            output_scale=output_scale,
        )
        return _SmState(
            t=x.t,
            u=None,
            ssv=ssv,
            num_data_points=state_previous.num_data_points,
        )

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        # todo: is this duplication unnecessary?
        # accepted = self._duplicate_with_unit_backward_model(s1)
        accepted = s1
        return InterpRes(accepted=accepted, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0 = self._interpolate_from_to_fn(
            rv=s0.ssv, output_scale=output_scale, t=t, t0=s0.t
        )
        posterior0 = _SmState(
            t=t,
            u=extrapolated0.extract_qoi(),
            # 'corrected' is the solution. We interpolate to get the value for
            # 'corrected' at time 't', which is exactly what happens.
            ssv=extrapolated0,
            num_data_points=s0.num_data_points,
        )

        extrapolated1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale=output_scale, t=s1.t, t0=t
        )
        backward_model1 = extrapolated1.backward_model
        ssv1 = self.extrapolation.replace_backward_model(
            s1.ssv, backward_model=backward_model1
        )

        t_accepted = jnp.maximum(s1.t, t)
        posterior1 = _SmState(
            t=t_accepted,
            u=s1.u,
            ssv=ssv1,
            num_data_points=s1.num_data_points,
        )

        return InterpRes(accepted=posterior1, solution=posterior0, previous=posterior0)

    # todo: move marginals to _SmState/FilterSol
    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior: MarkovSequence,
        posterior_previous: MarkovSequence,
        t0,
        t1,
        output_scale,
    ):
        acc, _sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, None, None, posterior),
            s0=self.init(t0, None, None, posterior_previous),
            output_scale=output_scale,
        )
        marginals = acc.backward_model.marginalise(marginals)
        u = marginals.extract_qoi()
        return u, marginals


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def _complete_extrapolation(
        self,
        state: _SmState,
        /,
        *,
        state_previous: _SmState,
        output_scale,
    ):
        ssv = self.extrapolation.complete_with_reversal(
            state.ssv,
            state_previous=state_previous.ssv,
            output_scale=output_scale,
        )

        merge_fn = state_previous.ssv.backward_model.merge_with_incoming_conditional
        bw_increment = ssv.backward_model
        backward_model = merge_fn(bw_increment)

        ssv = self.extrapolation.replace_backward_model(
            ssv, backward_model=backward_model
        )

        return _SmState(
            t=state.t,
            u=None,
            ssv=ssv,
            num_data_points=state_previous.num_data_points,
        )

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        merge_fn = s0.ssv.backward_model.merge_with_incoming_conditional
        backward_model1 = merge_fn(s1.ssv.backward_model)
        ssv1 = self.extrapolation.replace_backward_model(
            s1.ssv, backward_model=backward_model1
        )

        # Do we need:
        #  t_accepted = jnp.maximum(s1.t, t) ?
        solution = _SmState(
            t=t,
            u=s1.u,
            ssv=ssv1,
            num_data_points=s1.num_data_points,
        )

        # accepted = self._duplicate_with_unit_backward_model(solution)
        accepted = solution

        previous = accepted

        return InterpRes(accepted=accepted, solution=solution, previous=previous)

    def case_interpolate(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # From s0.t to t
        ssv0 = self._interpolate_from_to_fn(
            rv=s0.ssv,
            output_scale=output_scale,
            t=t,
            t0=s0.t,
        )
        bw0 = ssv0.backward_model
        backward_model0 = s0.ssv.backward_model.merge_with_incoming_conditional(bw0)
        ssv0 = self.extrapolation.replace_backward_model(
            ssv0, backward_model=backward_model0
        )

        solution = _SmState(
            t=t,
            u=ssv0.extract_qoi(),
            ssv=ssv0,
            num_data_points=s0.num_data_points,
        )

        # previous = self._duplicate_with_unit_backward_model(solution)
        previous = solution

        extra1 = self._interpolate_from_to_fn(
            rv=ssv0, output_scale=output_scale, t=s1.t, t0=t
        )
        backward_model1 = extra1.backward_model
        ssv1 = self.extrapolation.replace_backward_model(
            s1.ssv, backward_model=backward_model1
        )

        t_accepted = jnp.maximum(s1.t, t)
        accepted = _SmState(
            t=t_accepted,
            u=s1.u,
            ssv=ssv1,
            num_data_points=s1.num_data_points,
        )

        return InterpRes(accepted=accepted, solution=solution, previous=previous)

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior,
        posterior_previous: MarkovSequence,
        t0,
        t1,
        output_scale,
    ):
        raise NotImplementedError
