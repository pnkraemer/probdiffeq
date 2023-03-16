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
    >>> from jax.config import config
    >>> config.update("jax_platform_name", "cpu")

    >>> from probdiffeq import solvers
    >>> from probdiffeq.implementations import recipes
    >>> from probdiffeq.strategies import smoothers

    >>> print(generate_solver())
    MLESolver(strategy=Filter(implementation=<IsoTS0 with num_derivatives=4>))

    >>> print(generate_solver(num_derivatives=1))
    MLESolver(strategy=Filter(implementation=<IsoTS0 with num_derivatives=1>))

    >>> print(generate_solver(solver_factory=solvers.DynamicSolver))
    DynamicSolver(strategy=Filter(implementation=<IsoTS0 with num_derivatives=4>))

    >>> impl_fcty = recipes.DenseTS1.from_params
    >>> strat_fcty = smoothers.Smoother
    >>> print(generate_solver(strategy_factory=strat_fcty, impl_factory=impl_fcty, ode_shape=(1,)))  # noqa: E501
    MLESolver(strategy=Smoother(implementation=<DenseTS1 with num_derivatives=4>))
    """
    impl = impl_factory(**impl_factory_kwargs)
    strat = strategy_factory(impl)

    # I am not too happy with the need for this distinction below...

    if solver_factory in [solvers.MLESolver, solvers.DynamicSolver]:
        return solver_factory(strat)

    scale_sqrtm = impl.extrapolation.init_output_scale_sqrtm()
    return solver_factory(strat, output_scale_sqrtm=scale_sqrtm)
