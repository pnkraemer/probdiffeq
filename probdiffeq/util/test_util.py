"""Test utilities."""

from probdiffeq.solvers import calibrated
from probdiffeq.solvers.statespace import recipes
from probdiffeq.solvers.strategies import filters


def generate_solver(
    *,
    solver_factory=calibrated.mle,
    strategy_factory=filters.filter,
    impl_factory=recipes.ts0_iso,
    **impl_factory_kwargs,
):
    """Generate a solver.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax.config import config
    >>> config.update("jax_platform_name", "cpu")
    >>> jnp.set_printoptions(suppress=True, precision=2)  # summarise arrays

    >>> from probdiffeq.solvers import calibrated
    >>> from probdiffeq.solvers.statespace import recipes
    >>> from probdiffeq.solvers.strategies import smoothers

    >>> print(generate_solver())
    <MLE-solver with <Filter with <Isotropic IBM with num_derivatives=4>, <TS0 with ode_order=1>>>

    >>> print(generate_solver(num_derivatives=1))
    <MLE-solver with <Filter with <Isotropic IBM with num_derivatives=1>, <TS0 with ode_order=1>>>

    >>> print(generate_solver(solver_factory=calibrated.dynamic))
    <Dynamic solver with <Filter with <Isotropic IBM with num_derivatives=4>, <TS0 with ode_order=1>>>

    >>> impl_fcty = recipes.ts1_dense
    >>> strat_fcty = smoothers.smoother_adaptive
    >>> print(generate_solver(strategy_factory=strat_fcty, impl_factory=impl_fcty, ode_shape=(1,)))  # noqa: E501
    <MLE-solver with <Smoother with <Dense IBM with num_derivatives=4, ode_shape=(1,)>, <TS1 with ode_order=1>>>
    """
    implementation = impl_factory(**impl_factory_kwargs)
    strategy = strategy_factory(*implementation)
    return solver_factory(*strategy)
