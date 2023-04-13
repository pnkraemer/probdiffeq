"""''Global'' estimation: smoothing."""

import abc
import functools
from typing import Any, Generic, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _collections, _control_flow
from probdiffeq.strategies import _strategy

S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""

# todo: markov sequences should not necessarily be backwards


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[S]):
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

    def transform_unit_sample(self, base_sample, /):
        if base_sample.shape == self.sample_shape:
            return self._transform_one_unit_sample(base_sample)

        transform = self.transform_unit_sample
        transform_vmap = jax.vmap(transform, in_axes=0)
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


class _SmState(NamedTuple):
    t: Any
    extrapolated: Any

    corrected: Any

    backward_model: Any

    num_data_points: Any

    def scale_covariance(self, output_scale):
        bw_model = self.backward_model.scale_covariance(output_scale)
        if self.extrapolated is not None:
            # unexpectedly early call to scale_covariance...
            raise ValueError
        cor = self.corrected.scale_covariance(output_scale)
        return _SmState(
            t=self.t,
            extrapolated=None,
            corrected=cor,
            backward_model=bw_model,
            num_data_points=self.num_data_points,
        )


class _SmootherCommon(_strategy.Strategy):
    # Inherited abstract methods

    @abc.abstractmethod
    def case_interpolate(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_SmState]:
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_SmState]:
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
    def complete_extrapolation(
        self,
        output_extra: _SmState,
        /,
        *,
        output_scale,
        state_previous: _SmState,
    ):
        raise NotImplementedError

    def init(self, t, posterior, /) -> _SmState:
        return _SmState(
            t=t,
            corrected=posterior.init,
            extrapolated=None,
            backward_model=posterior.backward_model,
            num_data_points=posterior.num_data_points,
        )

    def solution_from_tcoeffs(self, taylor_coefficients, *, num_data_points):
        corrected = self.extrapolation.init_state_space_var(
            taylor_coefficients=taylor_coefficients
        )

        init_bw_model = self.extrapolation.init_conditional
        bw_model = init_bw_model(ssv_proto=corrected)
        return MarkovSequence(
            init=corrected, backward_model=bw_model, num_data_points=num_data_points
        )

    def extract(self, state: _SmState, /) -> MarkovSequence:
        return MarkovSequence(
            init=state.corrected,
            backward_model=state.backward_model,
            num_data_points=state.num_data_points,
        )

    def begin_extrapolation(self, posterior: _SmState, /, *, dt) -> _SmState:
        ssv = self.extrapolation.begin_extrapolation(posterior.corrected, dt=dt)
        return _SmState(
            t=posterior.t + dt,
            extrapolated=ssv,
            corrected=None,
            backward_model=None,
            num_data_points=posterior.num_data_points,
        )

    def begin_correction(
        self, output_extra: _SmState, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, Any]:
        return self.correction.begin_correction(
            output_extra.extrapolated, vector_field=vector_field, t=t, p=p
        )

    def complete_correction(self, extrapolated: _SmState, /, *, cache_obs):
        a, corrected = self.correction.complete_correction(
            extrapolated=extrapolated.extrapolated, cache=cache_obs
        )
        corrected_seq = _SmState(
            t=extrapolated.t,
            corrected=corrected,
            extrapolated=None,  # not relevant anymore
            backward_model=extrapolated.backward_model,
            num_data_points=extrapolated.num_data_points + 1,
        )

        return a, corrected_seq

    def extract_u(self, *, state: _SmState):
        return state.corrected.extract_qoi()

    def extract_marginals_terminal_values(self, posterior: MarkovSequence, /):
        return posterior.init

    def extract_marginals(self, posterior: MarkovSequence, /):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        markov = MarkovSequence(
            init=init,
            backward_model=posterior.backward_model,
            num_data_points=posterior.num_data_points,
        )
        return markov.marginalise_backwards()

    def sample(self, key, *, posterior: MarkovSequence, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        base_samples = jax.random.normal(key=key, shape=shape + posterior.sample_shape)
        return posterior.transform_unit_sample(base_samples)

    def num_data_points(self, state, /):
        return state.num_data_points

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale, t, t0):
        # todo: act on state instead of rv+t0
        dt = t - t0
        output_extra = self.extrapolation.begin_extrapolation(rv, dt=dt)

        _extra = self.extrapolation
        extra_fn = _extra.complete_extrapolation_with_reversal
        extrapolated, bw_model = extra_fn(
            output_extra,
            s0=rv,
            output_scale=output_scale,
        )
        return extrapolated, bw_model  # should this return a MarkovSequence?

    # todo: should this be a classmethod of MarkovSequence?
    def _duplicate_with_unit_backward_model(self, posterior: _SmState, /) -> _SmState:
        init_conditional_fn = self.extrapolation.init_conditional
        bw_model = init_conditional_fn(ssv_proto=posterior.corrected)
        return _SmState(
            t=posterior.t,
            extrapolated=posterior.extrapolated,
            corrected=posterior.corrected,
            backward_model=bw_model,
            num_data_points=posterior.num_data_points,
        )


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self,
        output_extra: _SmState,
        /,
        *,
        output_scale,
        state_previous: _SmState,
    ) -> _SmState:
        extra = self.extrapolation
        extra_fn = extra.complete_extrapolation_with_reversal
        extrapolated, bw_model = extra_fn(
            output_extra.extrapolated,
            s0=state_previous.corrected,
            output_scale=output_scale,
        )
        return _SmState(
            t=output_extra.t,
            extrapolated=extrapolated,
            corrected=None,
            backward_model=bw_model,
            num_data_points=state_previous.num_data_points,
        )

    def case_right_corner(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # todo: is this duplication unnecessary?
        accepted = self._duplicate_with_unit_backward_model(s1)
        return _collections.InterpRes(accepted=accepted, solution=s1, previous=s1)

    def case_interpolate(
        self, *, s0: _SmState, s1: _SmState, t, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=s0.corrected, output_scale=output_scale, t=t, t0=s0.t
        )
        posterior0 = _SmState(
            t=t,
            # 'corrected' is the solution. We interpolate to get the value for
            # 'corrected' at time 't', which is exactly what happens.
            extrapolated=None,
            corrected=extrapolated0,
            backward_model=backward_model0,
            num_data_points=s0.num_data_points,
        )

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale=output_scale, t=s1.t, t0=t
        )
        t_accepted = jnp.maximum(s1.t, t)
        posterior1 = _SmState(
            t=t_accepted,
            extrapolated=s1.extrapolated,  # None
            corrected=s1.corrected,
            backward_model=backward_model1,
            num_data_points=s1.num_data_points,
        )

        return _collections.InterpRes(
            accepted=posterior1, solution=posterior0, previous=posterior0
        )

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
            s1=self.init(t1, posterior),
            s0=self.init(t0, posterior_previous),
            # t0=t0,
            # t1=t1,
            output_scale=output_scale,
        )
        marginals_at_t = acc.backward_model.marginalise(marginals)
        u = marginals_at_t.extract_qoi()
        return u, marginals_at_t


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def complete_extrapolation(
        self,
        output_extra: _SmState,
        /,
        *,
        state_previous: _SmState,
        output_scale,
    ):
        _temp = self.extrapolation.complete_extrapolation_with_reversal(
            output_extra.extrapolated,
            s0=state_previous.corrected,
            output_scale=output_scale,
        )
        extrapolated, bw_increment = _temp

        merge_fn = state_previous.backward_model.merge_with_incoming_conditional
        backward_model = merge_fn(bw_increment)

        return _SmState(
            t=output_extra.t,
            extrapolated=extrapolated,
            corrected=None,
            backward_model=backward_model,
            num_data_points=state_previous.num_data_points,
        )

    def case_right_corner(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        merge_fn = s0.backward_model.merge_with_incoming_conditional
        backward_model1 = merge_fn(s1.backward_model)

        solution = _SmState(
            t=t,
            extrapolated=s1.extrapolated,
            corrected=s1.corrected,
            backward_model=backward_model1,
            num_data_points=s1.num_data_points,
        )

        accepted = self._duplicate_with_unit_backward_model(solution)
        previous = accepted

        return _collections.InterpRes(
            accepted=accepted, solution=solution, previous=previous
        )

    def case_interpolate(
        self, *, s0: _SmState, s1: _SmState, t, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.corrected,
            output_scale=output_scale,
            t=t,
            t0=s0.t,
        )
        backward_model0 = s0.backward_model.merge_with_incoming_conditional(bw0)
        solution = _SmState(
            t=t,
            extrapolated=None,
            corrected=extrapolated0,
            backward_model=backward_model0,
            num_data_points=s0.num_data_points,
        )

        previous = self._duplicate_with_unit_backward_model(solution)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale=output_scale, t=s1.t, t0=t
        )
        t_accepted = jnp.maximum(s1.t, t)
        accepted = _SmState(
            t=t_accepted,
            extrapolated=s1.extrapolated,  # None
            corrected=s1.corrected,
            backward_model=backward_model1,
            num_data_points=s1.num_data_points,
        )

        return _collections.InterpRes(
            accepted=accepted, solution=solution, previous=previous
        )

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
