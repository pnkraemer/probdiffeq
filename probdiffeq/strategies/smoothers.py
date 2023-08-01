"""''Global'' estimation: smoothing."""

import jax
import jax.numpy as jnp

from probdiffeq import _interp, _markov
from probdiffeq.strategies import _strategy


def smoother(*impl):
    """Create a smoother strategy."""
    extra, corr, calib = impl
    return _Smoother(extra.smoother, corr), calib


def smoother_fixedpoint(*impl):
    """Create a fixedpoint-smoother strategy."""
    extra, corr, calib = impl
    return _FixedPointSmoother(extra.fixedpoint, corr), calib


@jax.tree_util.register_pytree_node_class
class _Smoother(_strategy.Strategy):
    """Smoother."""

    def case_right_corner(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        return _smoother_interpolate(
            t, s0=s0, s1=s1, output_scale=output_scale, extrapolation=self.extrapolation
        )

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior: _markov.MarkovSeqRev,
        posterior_previous: _markov.MarkovSeqRev,
        t0,
        t1,
        output_scale,
    ):
        return _smoother_offgrid_marginals(
            t,
            marginals=marginals,
            output_scale=output_scale,
            posterior=posterior,
            posterior_previous=posterior_previous,
            t0=t0,
            t1=t1,
            init=self.init,
            extract=self.extract,
            interpolate=self.case_interpolate,
        )


def _smoother_offgrid_marginals(
    t,
    *,
    marginals,
    output_scale,
    posterior,
    posterior_previous,
    t0,
    t1,
    init,
    extract,
    interpolate,
):
    acc, _sol, _prev = interpolate(
        t=t,
        s1=init(t1, posterior),
        s0=init(t0, posterior_previous),
        output_scale=output_scale,
    )
    t, posterior = extract(acc)
    marginals = posterior.conditional.marginalise(marginals)
    u = marginals.extract_qoi_from_sample(marginals.mean)
    return u, marginals


# todo: state_t0 and state_t1 variable names
def _smoother_interpolate(t, *, s0, s1, output_scale, extrapolation):
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
    e_t = _extrapolate(
        s0=s0, output_scale=output_scale, t=t, extrapolation=extrapolation
    )
    e_1 = _extrapolate(
        s0=e_t, output_scale=output_scale, t=s1.t, extrapolation=extrapolation
    )
    # Marginalise from t1 to t to obtain the interpolated solution.
    # (This function assumes we are in the forward-pass, which is why we interpolate
    # back from the "filtering" state. In the context of offgrid_marginals,
    # the backward-interpolation step is repeated from the smoothing marginals)
    bw_t1_to_t, bw_t_to_t0 = e_1.extra, e_t.extra
    rv_at_t = bw_t1_to_t.marginalise(s1.ssv.hidden_state)
    mseq_t = _markov.MarkovSeqRev(init=rv_at_t, conditional=bw_t_to_t0)
    ssv, _ = extrapolation.init(mseq_t)
    corr_like = jax.tree_util.tree_map(jnp.empty_like, s1.corr)
    state_at_t = _strategy.State(t=t, ssv=ssv, corr=corr_like, extra=bw_t_to_t0)
    # The state at t1 gets a new backward model; it must remember how to
    # get back to t, not to t0.
    # The other two are the extrapolated solution
    s_1 = _strategy.State(t=s1.t, ssv=s1.ssv, corr=s1.corr, extra=bw_t1_to_t)
    return _interp.InterpRes(accepted=s_1, solution=state_at_t, previous=state_at_t)


@jax.tree_util.register_pytree_node_class
class _FixedPointSmoother(_strategy.Strategy):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def case_right_corner(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ):
        return _fixedpoint_right_corner(
            t, s0=s0, s1=s1, output_scale=output_scale, extrapolation=self.extrapolation
        )

    def case_interpolate(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        return _fixedpoint_interpolate(
            t, s0=s0, s1=s1, output_scale=output_scale, extrapolation=self.extrapolation
        )


def _fixedpoint_right_corner(t, *, s0, s1, output_scale, extrapolation):
    # See case_interpolate() for detailed explanation of why this works.
    # Todo: this prepares _future_ steps, so shouldn't it happen
    #  at initialisation instead of at completion?
    ssv, extra = extrapolation.reset(s1.ssv, s1.extra)
    accepted = _strategy.State(t=s1.t, ssv=ssv, extra=extra, corr=s1.corr)
    previous = _strategy.State(t=s1.t, ssv=ssv, extra=extra, corr=s1.corr)
    return _interp.InterpRes(accepted=accepted, solution=s1, previous=previous)


def _fixedpoint_interpolate(t, *, s0, s1, output_scale, extrapolation):
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
    e_t = _extrapolate(
        s0=s0, output_scale=output_scale, t=t, extrapolation=extrapolation
    )

    ssv, extra = extrapolation.reset(e_t.ssv, e_t.extra)
    prev_t = _strategy.State(t=e_t.t, ssv=ssv, extra=extra, corr=e_t.corr)

    e_1 = _extrapolate(
        s0=prev_t,
        output_scale=output_scale,
        t=s1.t,
        extrapolation=extrapolation,
    )

    # Marginalise from t1 to t:
    # turn an extrapolation- ("e_t") into an interpolation-result ("i_t")
    # Note how we use the bw_to_to_qoi backward model!
    # (Which is different for the non-fixed-point smoother)
    bw_t1_to_t, bw_t_to_qoi = e_1.extra, e_t.extra
    rv_t = bw_t1_to_t.marginalise(s1.ssv.hidden_state)
    mseq_t = _markov.MarkovSeqRev(init=rv_t, conditional=bw_t_to_qoi)
    ssv_t, _ = extrapolation.init(mseq_t)
    corr_like = jax.tree_util.tree_map(jnp.empty_like, s1.corr)
    sol_t = _strategy.State(t=t, ssv=ssv_t, corr=corr_like, extra=bw_t_to_qoi)

    # Future IVP solver stepping continues from here:
    acc_t1 = _strategy.State(t=s1.t, ssv=s1.ssv, corr=s1.corr, extra=bw_t1_to_t)

    # Bundle up the results and return
    return _interp.InterpRes(accepted=acc_t1, solution=sol_t, previous=prev_t)


def _extrapolate(s0, output_scale, t, *, extrapolation):
    dt = t - s0.t
    ssv, extra = extrapolation.begin(s0.ssv, s0.extra, dt=dt)
    ssv, extra = extrapolation.complete(ssv, extra, output_scale=output_scale)
    corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
    return _strategy.State(t=t, ssv=ssv, extra=extra, corr=corr_like)
