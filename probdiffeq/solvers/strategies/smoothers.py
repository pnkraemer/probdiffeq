"""''Global'' estimation: smoothing."""


from probdiffeq.solvers.strategies import strategy


def smoother_adaptive(extrapolation_factory, correction, /):
    """Create a smoother strategy."""
    extrapolation = extrapolation_factory.dense()
    extrapolation_repr = extrapolation_factory.string_repr()
    return strategy.Strategy(
        extrapolation,
        correction,
        is_suitable_for_save_at=False,
        is_suitable_for_offgrid_marginals=True,
        string_repr=f"<Smoother with {extrapolation_repr}, {correction}>",
    )


def fixedpoint_adaptive(extrapolation_factory, correction, /):
    """Create a fixedpoint-smoother strategy."""
    extrapolation = extrapolation_factory.save_at()
    extrapolation_repr = extrapolation_factory.string_repr()
    return strategy.Strategy(
        extrapolation,
        correction,
        is_suitable_for_save_at=True,
        is_suitable_for_offgrid_marginals=False,
        string_repr=f"<Fixed-point smoother with {extrapolation_repr}, {correction}>",
    )
