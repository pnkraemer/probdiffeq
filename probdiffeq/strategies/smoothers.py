"""''Global'' estimation: smoothing."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _interp, _markov
from probdiffeq.backend import containers
from probdiffeq.strategies import _strategy


class _SmState(containers.NamedTuple):
    t: Any
    ssv: Any
    extra: Any

    corr: Any

    @property
    def u(self):
        return self.ssv.u

    def scale_covariance(self, output_scale):
        extra = self.extra.scale_covariance(output_scale)
        ssv = self.ssv.scale_covariance(output_scale)
        corr = self.corr.scale_covariance(output_scale)
        return _SmState(t=self.t, extra=extra, ssv=ssv, corr=corr)


@jax.tree_util.register_pytree_node_class
class Smoother(_strategy.Strategy):
    """Smoother."""

    def complete(self, state, /, *, output_scale, vector_field, parameters):
        ssv, extra = self.extrapolation.smoother.complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(t=state.t, corr=corr, extra=extra, ssv=ssv)

    def case_right_corner(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> _interp.InterpRes[_SmState]:
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> _interp.InterpRes[_SmState]:
        """Interpolate.

        A smoother interpolates by_
        * Extrapolating from t0 to t, which gives the "filtering" marginal
          and the backward transition from t to t0.
        * Extrapolating from t to t1, which gives another "filtering" marginal
          and the backward transition from t1 to t.
        * Applying the t1-to-t backward transition to compute the interpolation result.
          This intermediate result is informed about its "right-hand side" datum.

        Subsequent interpolations continue from the value at 't'.
        Subsequent IVP solver steps continue from the value at 't1'.
        """
        # Extrapolate from t0 to t, and from t to t1. This yields all building blocks.
        e_t = self._extrapolate(s0=s0, output_scale=output_scale, t=t)
        e_1 = self._extrapolate(s0=e_t, output_scale=output_scale, t=s1.t)

        # Marginalise from t1 to t to obtain the interpolated solution.
        bw_t1_to_t, bw_t_to_t0 = e_1.extra, e_t.extra
        rv_at_t = bw_t1_to_t.marginalise(s1.ssv.hidden_state)
        mseq_t = _markov.MarkovSequence(init=rv_at_t, backward_model=bw_t_to_t0)
        ssv, _ = self.extrapolation.smoother.init(mseq_t)
        corr_like = jax.tree_util.tree_map(jnp.empty_like, s1.corr)
        state_at_t = _SmState(t=t, ssv=ssv, corr=corr_like, extra=bw_t_to_t0)

        # The state at t1 gets a new backward model; it must remember how to
        # get back to t, not to t0.
        # The other two are the extrapolated solution
        s_1 = _SmState(t=s1.t, ssv=s1.ssv, corr=s1.corr, extra=bw_t1_to_t)
        return _interp.InterpRes(accepted=s_1, solution=state_at_t, previous=state_at_t)

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior: _markov.MarkovSequence,
        posterior_previous: _markov.MarkovSequence,
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
        marginals = posterior.backward_model.marginalise(marginals)
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals

    def init(self, t, posterior, /) -> _SmState:
        ssv, extra = self.extrapolation.smoother.init(posterior)
        ssv, corr = self.correction.init(ssv)
        return _SmState(t=t, ssv=ssv, extra=extra, corr=corr)

    def begin(self, state: _SmState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.smoother.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(t=state.t + dt, ssv=ssv, extra=extra, corr=corr)

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        sol = self.extrapolation.smoother.solution_from_tcoeffs(taylor_coefficients)
        marginals = sol.init
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, state: _SmState, /) -> Tuple[float, _markov.MarkovSequence]:
        ssv = self.correction.extract(state.ssv, state.corr)
        sol = self.extrapolation.smoother.extract(ssv, state.extra)
        return state.t, sol

    def _extrapolate(self, *, s0, output_scale, t):
        dt = t - s0.t
        ssv, extra = self.extrapolation.smoother.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.smoother.complete(
            ssv, extra, output_scale=output_scale
        )
        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        return _SmState(t=t, ssv=ssv, extra=extra, corr=corr_like)

    def promote_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.smoother.promote_output_scale
        return init_fn(*args, **kwargs)

    def extract_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.smoother.extract_output_scale
        return init_fn(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_strategy.Strategy):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        seq = self.extrapolation.fixedpoint.solution_from_tcoeffs(taylor_coefficients)
        marginals = seq.init
        u = taylor_coefficients[0]
        return u, marginals, seq

    def init(self, t, posterior, /) -> _SmState:
        ssv, extra = self.extrapolation.fixedpoint.init(posterior)
        ssv, corr = self.correction.init(ssv)
        return _SmState(t=t, ssv=ssv, extra=extra, corr=corr)

    def begin(self, state: _SmState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.fixedpoint.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(t=state.t + dt, ssv=ssv, extra=extra, corr=corr)

    def complete(self, state, /, *, output_scale, vector_field, parameters):
        ssv, extra = self.extrapolation.fixedpoint.complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        bw_model, *_ = state.extra
        extra = bw_model.merge_with_incoming_conditional(extra)
        ssv, corr = self.correction.complete(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _SmState(t=state.t, corr=corr, extra=extra, ssv=ssv)

    def case_right_corner(self, t, *, s0: _SmState, s1: _SmState, output_scale):
        # See case_interpolate() for detailed explanation of why this works.

        # todo: should reset_fixedpoint see both s1 and s0, and return
        #  the merged & the reset' versions?
        # todo: is this merging even necessary?
        bw_t_to_qoi = s0.extra.merge_with_incoming_conditional(s1.extra)
        solution = _SmState(t=t, ssv=s1.ssv, corr=s1.corr, extra=bw_t_to_qoi)

        accepted = self._reset_fixedpoint(solution)
        previous = self._reset_fixedpoint(solution)
        return _interp.InterpRes(
            accepted=accepted, solution=solution, previous=previous
        )

    def case_interpolate(
        self, t, *, s0: _SmState, s1: _SmState, output_scale
    ) -> _interp.InterpRes[_SmState]:
        """Interpolate.

        A fixed-point smoother interpolates by

        * Extrapolating from t0 to t, which gives the "filtering" marginal
          and the backward transition from t to t0.
        * Extrapolating from t to t1, which gives another "filtering" marginal
          and the backward transition from t1 to t.
        * Applying the t1-to-t backward transition to compute the interpolation result.
          This intermediate result is informed about its "right-hand side" datum.

        The difference to smoother-interpolation is quite subtle:

        * The backward transition of the value at 't' is merged with that at 't0'.
          The reason is that the backward transition at 't0' knows
          "how to get to the quantity of interest",
          and this is precisely what we want to interpolate.
        * Subsequent interpolations do not continue from the value at 't', but
          from a very similar value where the backward transition
          is replaced with an identity. The reason is that the interpolated solution
          becomes the new quantity of interest, and subsequent interpolations
          need to learn how to get here.
        * Subsequent solver steps do not continue from the value at 't1',
          but the value at 't1' where the backward model is replaced by
          the 't1-to-t' backward model. The reason is similar to the above:
          future steps need to know "how to get back to the quantity of interest",
          which is the interpolated solution.

        These distinctions are precisely why we need three fields
        in every interpolation result:
        the solution,
        the continue-interpolation-from-here,
        and the continue-stepping-from-here.
        All three are different for fixed point smoothers.
        (Really, I try removing one of them monthly and
        then don't understand why tests fail.)
        """
        # Extrapolate from t0 to t, and from t to t1. This yields all building blocks.
        # No backward model condensing yet.
        # 'e_t': interpolated result at time 't'.
        # 'e_1': extrapolated result at time 't1'.
        # todo: rename this to "extrapolate from to fn", no interpolation happens here.
        e_t = self._extrapolate(s0=s0, output_scale=output_scale, t=t)
        prev_t = self._reset_fixedpoint(e_t)
        e_1 = self._extrapolate(s0=prev_t, output_scale=output_scale, t=s1.t)

        # Read backward models and condense qoi-to-t
        bw_t1_to_t, bw_t_to_t0, bw_t0_to_qoi = e_1.extra, e_t.extra, s0.extra
        bw_t_to_qoi = bw_t0_to_qoi.merge_with_incoming_conditional(bw_t_to_t0)

        # Marginalise from t1 to t:
        # turn an extrapolation- ("e_t") into an interpolation-result ("i_t")
        # Note how we use the bw_to_to_qoi backward model!
        # (Which is different for the non-fixed-point smoother)
        rv_t = bw_t1_to_t.marginalise(s1.ssv.hidden_state)
        mseq_t = _markov.MarkovSequence(init=rv_t, backward_model=bw_t_to_qoi)
        ssv_t, _ = self.extrapolation.fixedpoint.init(mseq_t)
        corr_like = jax.tree_util.tree_map(jnp.empty_like, s1.corr)
        sol_t = _SmState(t=t, ssv=ssv_t, corr=corr_like, extra=bw_t_to_qoi)

        # Future IVP solver stepping continues from here:
        acc_t1 = _SmState(t=s1.t, ssv=s1.ssv, corr=s1.corr, extra=bw_t1_to_t)

        # Bundle up the results and return
        return _interp.InterpRes(accepted=acc_t1, solution=sol_t, previous=prev_t)

    def extract(self, state: _SmState, /) -> Tuple[float, _markov.MarkovSequence]:
        ssv = self.correction.extract(state.ssv, state.corr)
        sol = self.extrapolation.fixedpoint.extract(ssv, state.extra)
        return state.t, sol

    # Auxiliary routines that are the same among all subclasses

    def _extrapolate(self, *, s0, output_scale, t):
        dt = t - s0.t
        ssv, extra = self.extrapolation.fixedpoint.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.fixedpoint.complete(
            ssv, extra, output_scale=output_scale
        )
        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        return _SmState(t=t, ssv=ssv, extra=extra, corr=corr_like)

    # todo: move this to (a new) self.extrapolation.fixedpoint.reset()
    def _reset_fixedpoint(self, state: _SmState, /) -> _SmState:
        extra_new = self.extrapolation.fixedpoint.init_conditional(
            rv_proto=state.extra.noise
        )
        return _SmState(t=state.t, extra=extra_new, corr=state.corr, ssv=state.ssv)

    def promote_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.fixedpoint.promote_output_scale
        return init_fn(*args, **kwargs)

    def extract_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.fixedpoint.extract_output_scale
        return init_fn(*args, **kwargs)
