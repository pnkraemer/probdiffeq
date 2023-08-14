"""''Global'' estimation: smoothing."""


from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import _common, strategy


def smoother_adaptive(extrapolation_factory, corr, /):
    """Create a smoother strategy."""
    extrapolation = extrapolation_factory.dense()
    extrapolation_repr = extrapolation_factory.string_repr()
    return strategy.Strategy(
        extrapolation,
        corr,
        is_suitable_for_save_at=False,
        string_repr=f"<Smoother with {extrapolation_repr}, {corr}>",
        # Right-corner: use default
        impl_right_corner="default",
        # Interpolate like a smoother:
        impl_offgrid_marginals=_smoother_offgrid_marginals,
    )


def fixedpoint_adaptive(extrapolation_factory, corr, /):
    """Create a fixedpoint-smoother strategy."""
    extrapolation = extrapolation_factory.save_at()
    extrapolation_repr = extrapolation_factory.string_repr()
    return strategy.Strategy(
        extrapolation,
        corr,
        is_suitable_for_save_at=True,
        string_repr=f"<Fixed-point with {extrapolation_repr}, {corr}>",
        # Offgrid-marginals are not available
        impl_offgrid_marginals=None,
        # Interpolate like a fixedpoint-smoother
        impl_right_corner=_fixedpoint_right_corner,
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
    marginals = impl.conditional.marginalise(marginals, posterior.conditional)
    u = impl.hidden_model.qoi(marginals)
    return u, marginals


def _fixedpoint_right_corner(state_at_t1, *, extrapolation):
    # See case_interpolate() for detailed explanation of why this works.
    # TODO: this prepares _future_ steps, so shouldn't it happen
    #  at initialisation instead of at completion?
    ssv, extra = extrapolation.reset(state_at_t1.hidden, state_at_t1.aux_extra)
    acc = _common.State(
        t=state_at_t1.t, hidden=ssv, aux_extra=extra, aux_corr=state_at_t1.aux_corr
    )
    pre = _common.State(
        t=state_at_t1.t, hidden=ssv, aux_extra=extra, aux_corr=state_at_t1.aux_corr
    )
    return _interp.InterpRes(accepted=acc, solution=state_at_t1, previous=pre)
