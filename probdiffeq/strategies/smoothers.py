"""''Global'' estimation: smoothing."""

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from probdiffeq._collections import InterpRes, MarkovSequence
from probdiffeq.strategies import _strategy


@jax.tree_util.register_pytree_node_class
class SmootherSol(_strategy.Posterior[MarkovSequence]):
    """Smmoothing solution."""

    def sample(self, key, *, shape):
        return self.rand.sample(key, shape=shape)

    def marginals_at_terminal_values(self):
        marginals = self.rand.init
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals

    def marginals(self):
        marginals = self._extract_marginals()
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals

    def _extract_marginals(self, /):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], self.rand.init)

        # todo: this construction should not happen here...
        markov = MarkovSequence(init=init, backward_model=self.rand.backward_model)
        return markov.marginalise_backwards()


class _SmState(NamedTuple):
    t: Any
    u: Any
    ssv: Any
    extra: Any

    corr: Any

    def scale_covariance(self, output_scale):
        return _SmState(
            t=self.t,
            u=self.u,
            extra=self.extra.scale_covariance(output_scale),
            ssv=self.ssv.scale_covariance(output_scale),
            corr=self.corr.scale_covariance(output_scale),
        )


@jax.tree_util.register_pytree_node_class
class Smoother(_strategy.Strategy):
    """Smoother."""

    def complete(self, state, /, *, output_scale, vector_field, parameters):
        ssv, extra = self.extrapolation.smoother_complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(
            t=state.t,
            u=ssv.extract_qoi(),
            corr=corr,
            extra=extra,
            ssv=ssv,
        )

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> InterpRes[_SmState]:
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
        s_t = self._interpolate_from_to_fn(s0=s0, output_scale=output_scale, t=t)
        state1 = self._interpolate_from_to_fn(s0=s_t, output_scale=output_scale, t=s1.t)
        backward_model1 = state1.extra

        s_1 = _SmState(
            t=s1.t,
            u=s1.u,
            ssv=s1.ssv,
            corr=s1.corr,
            extra=backward_model1,
        )
        return InterpRes(accepted=s_1, solution=s_t, previous=s_t)

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior: SmootherSol,
        posterior_previous: SmootherSol,
        t0,
        t1,
        output_scale,
    ):
        acc, _sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, posterior),
            s0=self.init(t0, posterior_previous),
            output_scale=output_scale,
        )
        t, posterior = self.extract(acc)
        marginals = posterior.rand.backward_model.marginalise(marginals)
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals

    def init(self, t, posterior, /) -> _SmState:
        ssv, extra = self.extrapolation.smoother_init(posterior.rand)
        ssv, corr = self.correction.init(ssv)
        return _SmState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=corr,
        )

    def begin(self, state: _SmState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.smoother_begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(
            t=state.t + dt,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=corr,
        )

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        seq = self.extrapolation.smoother_solution_from_tcoeffs(taylor_coefficients)
        sol = SmootherSol(seq)
        marginals = seq.init
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, state: _SmState, /) -> Tuple[float, SmootherSol]:
        ssv = self.correction.extract(state.ssv, state.corr)
        mseq = self.extrapolation.smoother_extract(ssv, state.extra)
        sol = SmootherSol(mseq)  # type: ignore
        return state.t, sol

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, s0, output_scale, t):
        dt = t - s0.t
        ssv, extra = self.extrapolation.smoother_begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.smoother_complete(
            ssv, extra, output_scale=output_scale
        )
        return _SmState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=jax.tree_util.tree_map(jnp.empty_like, s0.corr),
        )


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_strategy.Strategy):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def complete(self, state, /, *, output_scale, vector_field, parameters):
        ssv, extra = self.extrapolation.smoother_complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        bw_model, *_ = state.extra
        extra = bw_model.merge_with_incoming_conditional(extra)
        ssv, corr = self.correction.complete(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(
            t=state.t,
            u=ssv.extract_qoi(),
            corr=corr,
            extra=extra,
            ssv=ssv,
        )

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        backward_model1 = s0.extra.merge_with_incoming_conditional(s1.extra)

        solution = _SmState(
            t=t,
            u=s1.u,
            ssv=s1.ssv,
            corr=s1.corr,
            extra=backward_model1,
        )

        accepted = self._duplicate_with_unit_backward_model(solution)
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
        s_t = self._interpolate_from_to_fn(
            s0=s0,
            output_scale=output_scale,
            t=t,
        )
        extra = s0.extra.merge_with_incoming_conditional(s_t.extra)
        solution = _SmState(
            t=s_t.t,
            u=s_t.u,
            ssv=s_t.ssv,
            extra=extra,  # new
            corr=s_t.corr,
        )

        previous = self._duplicate_with_unit_backward_model(solution)

        s_1 = self._interpolate_from_to_fn(s0=s_t, output_scale=output_scale, t=s1.t)
        accepted = _SmState(
            t=s1.t,
            u=s1.u,
            ssv=s1.ssv,
            corr=s1.corr,
            extra=s_1.extra,  # new!
        )

        return InterpRes(accepted=accepted, solution=solution, previous=previous)

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior,
        posterior_previous: SmootherSol,
        t0,
        t1,
        output_scale,
    ):
        raise NotImplementedError

    def init(self, t, posterior, /) -> _SmState:
        ssv, extra = self.extrapolation.smoother_init(posterior.rand)
        ssv, corr = self.correction.init(ssv)
        return _SmState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=corr,
        )

    def begin(self, state: _SmState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.smoother_begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(
            t=state.t + dt,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=corr,
        )

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        seq = self.extrapolation.smoother_solution_from_tcoeffs(taylor_coefficients)
        sol = SmootherSol(seq)
        marginals = seq.init
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, state: _SmState, /) -> Tuple[float, SmootherSol]:
        ssv = self.correction.extract(state.ssv, state.corr)
        mseq = self.extrapolation.smoother_extract(ssv, state.extra)
        sol = SmootherSol(mseq)  # type: ignore
        return state.t, sol

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, s0, output_scale, t):
        dt = t - s0.t
        ssv, extra = self.extrapolation.smoother_begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.smoother_complete(
            ssv, extra, output_scale=output_scale
        )
        return _SmState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=jax.tree_util.tree_map(jnp.empty_like, s0.corr),
        )

    # todo: should this be a classmethod of MarkovSequence?
    def _duplicate_with_unit_backward_model(self, state: _SmState, /) -> _SmState:
        extra = self.extrapolation.smoother_init_conditional(rv_proto=state.extra.noise)
        return _SmState(
            t=state.t,
            u=state.u,
            extra=extra,  # new!
            corr=state.corr,
            ssv=state.ssv,
        )
