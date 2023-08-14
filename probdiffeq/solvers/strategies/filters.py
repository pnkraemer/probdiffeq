"""Forward-only estimation: filtering."""


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
        is_suitable_for_offgrid_marginals=True,
    )
