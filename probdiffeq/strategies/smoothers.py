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

    def __init__(self, *, init: S, backward_model):
        self.init = init
        self.backward_model = backward_model

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(init={self.init}, backward_model={self.backward_model})"

    def tree_flatten(self):
        children = (self.init, self.backward_model)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model = children
        return cls(init=init, backward_model=backward_model)

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
    ssv: Any
    backward_model: Any

    def scale_covariance(self, output_scale):
        bw_model = self.backward_model.scale_covariance(output_scale)
        ssv = self.ssv.scale_covariance(output_scale)
        return _SmState(ssv=ssv, backward_model=bw_model)


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

    def init(self, posterior, /) -> _SmState:
        return _SmState(ssv=posterior.init, backward_model=posterior.backward_model)

    def solution_from_tcoeffs(self, taylor_coefficients):
        corrected = self.implementation.extrapolation.init_state_space_var(
            taylor_coefficients=taylor_coefficients
        )

        init_bw_model = self.implementation.extrapolation.init_conditional
        bw_model = init_bw_model(ssv_proto=corrected)
        return MarkovSequence(init=corrected, backward_model=bw_model)

    def extract(self, state: _SmState, /) -> MarkovSequence:
        return MarkovSequence(init=state.ssv, backward_model=state.backward_model)

    def begin_extrapolation(self, posterior: _SmState, /, *, dt) -> _SmState:
        ssv = self.implementation.extrapolation.begin_extrapolation(
            posterior.ssv, dt=dt
        )
        return _SmState(ssv=ssv, backward_model=None)

    def begin_correction(
        self, output_extra: _SmState, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, Any]:
        ssv = output_extra.ssv
        return self.implementation.correction.begin_correction(
            ssv, vector_field=vector_field, t=t, p=p
        )

    def complete_correction(self, extrapolated: _SmState, /, *, cache_obs):
        a, (corrected, b) = self.implementation.correction.complete_correction(
            extrapolated=extrapolated.ssv, cache=cache_obs
        )
        corrected_seq = _SmState(
            ssv=corrected,
            backward_model=extrapolated.backward_model,
        )

        return a, (corrected_seq, b)

    def extract_u(self, posterior: MarkovSequence, /):
        return posterior.init.extract_qoi()

    def extract_marginals_terminal_values(self, posterior: MarkovSequence, /):
        return posterior.init

    def extract_marginals(self, posterior: MarkovSequence, /):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        markov = MarkovSequence(init=init, backward_model=posterior.backward_model)
        return markov.marginalise_backwards()

    def sample(self, key, *, posterior: MarkovSequence, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        base_samples = jax.random.normal(key=key, shape=shape + posterior.sample_shape)
        return posterior.transform_unit_sample(base_samples)

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale, t, t0):
        dt = t - t0
        output_extra = self.implementation.extrapolation.begin_extrapolation(rv, dt=dt)

        _extra = self.implementation.extrapolation
        extra_fn = _extra.complete_extrapolation_with_reversal
        extrapolated, bw_model = extra_fn(
            output_extra,
            s0=rv,
            output_scale=output_scale,
        )
        return extrapolated, bw_model  # should this return a MarkovSequence?

    # todo: should this be a classmethod of MarkovSequence?
    def _duplicate_with_unit_backward_model(self, posterior: _SmState, /) -> _SmState:
        init_conditional_fn = self.implementation.extrapolation.init_conditional
        bw_model = init_conditional_fn(ssv_proto=posterior.ssv)
        return _SmState(ssv=posterior.ssv, backward_model=bw_model)


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
        extra = self.implementation.extrapolation
        extra_fn = extra.complete_extrapolation_with_reversal
        extrapolated, bw_model = extra_fn(
            output_extra.ssv,
            s0=state_previous.ssv,
            output_scale=output_scale,
        )
        return _SmState(ssv=extrapolated, backward_model=bw_model)

    def case_right_corner(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # todo: is this duplication unnecessary?
        accepted = self._duplicate_with_unit_backward_model(s1)
        return _collections.InterpRes(accepted=accepted, solution=s1, previous=s1)

    def case_interpolate(
        self, *, s0: _SmState, s1: _SmState, t0, t1, t, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=s0.ssv, output_scale=output_scale, t=t, t0=t0
        )
        posterior0 = _SmState(ssv=extrapolated0, backward_model=backward_model0)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale=output_scale, t=t1, t0=t
        )
        posterior1 = _SmState(ssv=s1.ssv, backward_model=backward_model1)

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
            s1=self.init(posterior),
            s0=self.init(posterior_previous),
            t0=t0,
            t1=t1,
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
        _temp = self.implementation.extrapolation.complete_extrapolation_with_reversal(
            output_extra.ssv,
            s0=state_previous.ssv,
            output_scale=output_scale,
        )
        extrapolated, bw_increment = _temp

        merge_fn = state_previous.backward_model.merge_with_incoming_conditional
        backward_model = merge_fn(bw_increment)

        return _SmState(ssv=extrapolated, backward_model=backward_model)

    def case_right_corner(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        merge_fn = s0.backward_model.merge_with_incoming_conditional
        backward_model1 = merge_fn(s1.backward_model)

        solution = _SmState(ssv=s1.ssv, backward_model=backward_model1)

        accepted = self._duplicate_with_unit_backward_model(solution)
        previous = accepted

        return _collections.InterpRes(
            accepted=accepted, solution=solution, previous=previous
        )

    def case_interpolate(
        self, *, s0: _SmState, s1: _SmState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_SmState]:
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.ssv,
            output_scale=output_scale,
            t=t,
            t0=t0,
        )
        backward_model0 = s0.backward_model.merge_with_incoming_conditional(bw0)
        solution = _SmState(ssv=extrapolated0, backward_model=backward_model0)

        previous = self._duplicate_with_unit_backward_model(solution)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale=output_scale, t=t1, t0=t
        )
        accepted = _SmState(ssv=s1.ssv, backward_model=backward_model1)

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
