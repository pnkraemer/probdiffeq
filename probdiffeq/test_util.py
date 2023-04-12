"""Test utilities."""

from probdiffeq import ivpsolvers
from probdiffeq.ssm import recipes
from probdiffeq.strategies import filters


def generate_solver(
    *,
    solver_factory=ivpsolvers.MLESolver,
    strategy_factory=filters.Filter,
    impl_factory=recipes.ts0_iso,
    **impl_factory_kwargs,
):
    """Generate a solver.

    Examples
    --------
    >>> from jax.config import config
    >>> config.update("jax_platform_name", "cpu")

    >>> from probdiffeq import ivpsolvers
    >>> from probdiffeq.ssm import recipes
    >>> from probdiffeq.strategies import smoothers

    >>> print(generate_solver())
    MLESolver(strategy=Filter(implementation=<Implementation with num_derivatives=4, ode_order=1>))

    >>> print(generate_solver(num_derivatives=1))
    MLESolver(strategy=Filter(implementation=<Implementation with num_derivatives=1, ode_order=1>))

    >>> print(generate_solver(solver_factory=ivpsolvers.DynamicSolver))
    DynamicSolver(strategy=Filter(implementation=<Implementation with num_derivatives=4, ode_order=1>))

    >>> impl_fcty = recipes.ts1_dense
    >>> strat_fcty = smoothers.Smoother
    >>> print(generate_solver(strategy_factory=strat_fcty, impl_factory=impl_fcty, ode_shape=(1,)))  # noqa: E501
    MLESolver(strategy=Smoother(implementation=<Implementation with num_derivatives=4, ode_order=1>))
    """
    impl = impl_factory(**impl_factory_kwargs)
    strat = strategy_factory(impl)

    return solver_factory(strat)
    # # I am not too happy with the need for this distinction below...
    #
    # if solver_factory in [ivpsolvers.MLESolver, ivpsolvers.DynamicSolver]:
    #     return solver_factory(strat)
    #
    # output_scale = impl.extrapolation.init_output_scale()
    # return solver_factory(strat, output_scale=output_scale)
