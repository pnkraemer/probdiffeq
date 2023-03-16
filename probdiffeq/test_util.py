"""Test utilities."""
from probdiffeq import solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters


def generate_solver(
    *,
    solver_factory=solvers.MLESolver,
    strategy_factory=filters.Filter,
    impl_factory=recipes.IsoTS0.from_params,
    **impl_factory_kwargs,
):
    """Generate a solver.

    Examples
    --------
    >>> print(generate_solver())

    >>> print(generate_solver(num_derivatives=1))



    """
    impl = impl_factory(**impl_factory_kwargs)
    strat = strategy_factory(impl)
    return solver_factory(strat)
