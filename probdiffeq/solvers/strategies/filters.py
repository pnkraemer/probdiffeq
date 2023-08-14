"""Forward-only estimation: filtering."""


from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import strategy


def filter_adaptive(extrapolation_factory, correction, /):
    """Create a filter strategy."""
    extrapolation = extrapolation_factory.forward()
    extrapolation_repr = extrapolation_factory.string_repr()
    return strategy.Strategy(
        extrapolation,
        correction,
        string_repr=f"<Filter with {extrapolation_repr}, {correction}>",
        is_suitable_for_save_at=True,
        # Right-corner: use default
        impl_right_corner="default",
        # Filtering behaviour for interpolation
        impl_offgrid_marginals=_filter_offgrid_marginals,
    )


def _filter_offgrid_marginals(
    t,
    *,
    marginals,
    output_scale,
    posterior,
    posterior_previous,
    t0,
    t1,
    init,
    interpolate,
    extract,
):
    del marginals

    _acc, sol, _prev = interpolate(
        t=t,
        s1=init(t1, posterior),
        s0=init(t0, posterior_previous),
        output_scale=output_scale,
    )
    t, posterior = extract(sol)
    u = impl.hidden_model.qoi(posterior)
    return u, posterior
