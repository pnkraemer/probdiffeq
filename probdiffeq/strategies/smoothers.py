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
# todo: move markov sequence to state-space level
#  (solution type of smoother-style algos)


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
    ssv: Any
    extra: Any
    corr: Any

    # todo: are these properties a bit hacky?

    @property
    def t(self):
        return self.ssv.t

    @property
    def u(self):
        return self.ssv.u

    @property
    def error_estimate(self):
        return self.corr.error_estimate

    def scale_covariance(self, output_scale):
        ssv = self.ssv.scale_covariance(output_scale)
        extra = self.extra.scale_covariance(output_scale)
        corr = self.corr.scale_covariance(output_scale)
        return _SmState(ssv=ssv, extra=extra, corr=corr)


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

    def begin(self, state: _SmState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(ssv, state.corr, vector_field, parameters)
        return _SmState(ssv=ssv, extra=extra, corr=corr)

    @abc.abstractmethod
    def complete(self, state, /, *, vector_field, parameters, output_scale):
        raise NotImplementedError

    def solution_from_tcoeffs(self, taylor_coefficients, /, *, num_data_points):
        rv, cond = self.extrapolation.solution_from_tcoeffs_with_reversal(
            taylor_coefficients
        )
        sol = MarkovSequence(
            init=rv, backward_model=cond, num_data_points=num_data_points
        )
        marginals = rv
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, state: _SmState, /) -> _SolType:
        init, bw_model = self.extrapolation.extract_with_reversal(
            state.ssv, state.extra
        )
        markov_seq = MarkovSequence(
            init=init,
            backward_model=bw_model,
            num_data_points=state.ssv.num_data_points,
        )
        marginals = self._extract_marginals(markov_seq)
        u = state.ssv.extract_qoi_from_sample(marginals.mean)

        return state.t, u, marginals, markov_seq

    def extract_at_terminal_values(self, state: _SmState, /) -> _SolType:
        init, bw_model = self.extrapolation.extract_with_reversal(
            state.ssv, state.extra
        )
        markov_seq = MarkovSequence(
            init=init,
            backward_model=bw_model,
            num_data_points=state.ssv.num_data_points,
        )
        marginals = state.ssv.hidden_state
        u = state.ssv.extract_qoi_from_sample(marginals.mean)
        return state.t, u, marginals, markov_seq

    def _extract_marginals(self, posterior: MarkovSequence, /):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        markov = MarkovSequence(
            init=init,
            backward_model=posterior.backward_model,
            num_data_points=posterior.num_data_points,
        )
        return markov.marginalise_backwards()

    def num_data_points(self, state, /):
        return state.ssv.num_data_points

    def observation(self, state, /):
        return state.corr.observed

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_t0_to_t(self, *, state, output_scale, t, t0):
        # todo: act on state instead of rv+t0
        dt = t - t0
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, extra = self.extrapolation.complete_with_reversal(
            ssv, extra, output_scale=output_scale
        )
        return _SmState(
            ssv=ssv,
            extra=extra,
            corr=jax.tree_util.tree_map(jnp.empty_like, state.corr),
        )

    # # todo: should this be a classmethod of MarkovSequence?
    def _duplicate_with_unit_backward_model(self, posterior: _SmState, /) -> _SmState:
        # todo: this should be the init() method of a fixed-point extrapolation model
        #  once this is implemented, we can use simulate_terminal_values() between
        #  checkpoints and remove sooo much code.
        extra = self.extrapolation.duplicate_with_unit_backward_model(posterior.extra)
        return _SmState(ssv=posterior.ssv, extra=extra, corr=posterior.corr)


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def init(self, t, u, marginals, posterior, /) -> _SmState:
        # todo: should "extrapolation" see the "posterior" as a solution?
        #  I.e., should we move "MarkovSequence" to statespace.py?
        ssv, extra = self.extrapolation.init_with_reversal(
            t, u, posterior.init, posterior.backward_model, posterior.num_data_points
        )
        ssv, corr = self.correction.init(ssv)
        return _SmState(ssv=ssv, extra=extra, corr=corr)

    def complete(self, state, /, *, vector_field, parameters, output_scale):
        ssv, extra = self.extrapolation.complete_with_reversal(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(ssv, state.corr, vector_field, parameters)
        return _SmState(ssv=ssv, extra=extra, corr=corr)

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
        # todo: is this duplication necessary?
        # accepted = self._duplicate_with_unit_backward_model(s1)
        return InterpRes(accepted=s1, solution=s1, previous=s1)

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
        posterior0 = self._interpolate_from_t0_to_t(
            state=s0, output_scale=output_scale, t=t, t0=s0.t
        )
        # posterior0 = _SmState(
        #     t=t,
        #     u=extrapolated0.extract_qoi(),
        #     ssv=extrapolated0,
        #     num_data_points=s0.num_data_points,
        # )

        posterior1 = self._interpolate_from_t0_to_t(
            state=posterior0, output_scale=output_scale, t=s1.t, t0=t
        )
        # t_accepted = jnp.maximum(s1.t, t)
        posterior1 = _SmState(
            ssv=s1.ssv,
            corr=s1.corr,
            # replace backward model:
            extra=posterior1.extra,
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
        acc_p, _sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, None, None, posterior),
            s0=self.init(t0, None, None, posterior_previous),
            output_scale=output_scale,
        )
        _, _, _, acc = self.extract_at_terminal_values(acc_p)
        marginals = acc.backward_model.marginalise(marginals)
        u = acc_p.ssv.extract_qoi_from_sample(marginals.mean)
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

    def init(self, t, u, marginals, posterior, /) -> _SmState:
        ssv, extra = self.extrapolation.init_with_reversal_and_reset(
            t, u, posterior.init, posterior.backward_model, posterior.num_data_points
        )
        ssv, corr = self.correction.init(ssv)
        return _SmState(ssv=ssv, extra=extra, corr=corr)

    def complete(self, state, /, *, vector_field, parameters, output_scale):
        bw_previous = state.extra.backward_model
        ssv, extra = self.extrapolation.complete_with_reversal(
            state.ssv, state.extra, output_scale=output_scale
        )
        # now this is something that should really happen in extrapolate().
        backward_model = bw_previous.merge_with_incoming_conditional(
            extra.backward_model
        )
        extra = self.extrapolation.replace_backward_model(
            extra, backward_model=backward_model
        )

        ssv, corr = self.correction.complete(ssv, state.corr, vector_field, parameters)
        return _SmState(ssv=ssv, extra=extra, corr=corr)

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        merge_fn = s0.extra.backward_model.merge_with_incoming_conditional
        backward_model1 = merge_fn(s1.extra.backward_model)
        extra1 = self.extrapolation.replace_backward_model(
            s1.extra, backward_model=backward_model1
        )

        # Do we need:
        #  t_accepted = jnp.maximum(s1.t, t) ?
        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        solution = _SmState(ssv=s1.ssv, extra=extra1, corr=corr_like)

        accepted = self._duplicate_with_unit_backward_model(solution)
        # accepted = solution
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
        posterior0 = self._interpolate_from_t0_to_t(
            state=s0,
            output_scale=output_scale,
            t=t,
            t0=s0.t,
        )
        backward_model0 = s0.extra.backward_model.merge_with_incoming_conditional(
            posterior0.extra.backward_model
        )
        extra0 = self.extrapolation.replace_backward_model(
            posterior0.extra, backward_model=backward_model0
        )

        solution = _SmState(ssv=posterior0.ssv, corr=posterior0.corr, extra=extra0)
        previous = self._duplicate_with_unit_backward_model(solution)

        posterior1 = self._interpolate_from_t0_to_t(
            state=posterior0, output_scale=output_scale, t=s1.t, t0=t
        )
        backward_model1 = posterior1.extra.backward_model
        extra1 = self.extrapolation.replace_backward_model(
            s1.extra, backward_model=backward_model1
        )

        # t_accepted = jnp.maximum(s1.t, t)
        accepted = _SmState(ssv=s1.ssv, extra=extra1, corr=posterior1.corr)

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
