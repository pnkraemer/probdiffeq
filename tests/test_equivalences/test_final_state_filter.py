"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve_with_python_while_loop() calls with default smoothers.
import jax
import jax.numpy as jnp

from probdiffeq import controls, ivpsolve, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.case
def strategy_pair_smoother():
    impl = recipes.ts0_iso()
    return filters.Filter(*impl), smoothers.Smoother(*impl)


@testing.case
def strategy_pair_fixedpoint_smoother():
    impl = recipes.ts0_iso()
    return filters.Filter(*impl), smoothers.FixedPointSmoother(*impl)


@testing.parametrize_with_cases("fil, smo", cases=".", prefix="strategy_pair_")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize(
    "ctrl", [controls.IntegralClipped(), controls.ProportionalIntegralClipped()]
)
def test_final_state_equal_to_filter_clipped_control(ode_problem, fil, smo, ctrl):
    """Assert equality for when a clipped controller is used."""
    atol, rtol = 1e-2, 1e-1
    filter_solution = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=ivpsolvers.DynamicSolver(strategy=fil),
        atol=atol,
        rtol=rtol,
        control=ctrl,
    )
    smoother_solution = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=ivpsolvers.DynamicSolver(strategy=smo),
        atol=atol,
        rtol=rtol,
        control=ctrl,
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    assert _tree_all_allclose(filter_solution.t, smoother_solution.t)
    assert _tree_all_allclose(filter_solution.u, smoother_solution.u)

    filter_scale = filter_solution.output_scale
    smoother_scale = smoother_solution.output_scale
    assert _tree_all_allclose(filter_scale, smoother_scale)

    filter_marginals = filter_solution.marginals
    smoother_marginals = smoother_solution.marginals
    assert _tree_all_allclose(filter_marginals.mean, smoother_marginals.mean)

    filter_cov_sqrtm = filter_marginals.cov_sqrtm_lower
    smoother_cov_sqrtm = smoother_marginals.cov_sqrtm_lower
    assert _tree_all_allclose(cov(filter_cov_sqrtm), cov(smoother_cov_sqrtm))


@testing.parametrize_with_cases("fil, smo", cases=".", prefix="strategy_pair_")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize("ctrl", [controls.Integral(), controls.ProportionalIntegral()])
def test_final_state_not_equal_to_filter_nonclipped_control(
    ode_problem, fil, smo, ctrl
):
    """Assert inequality for when a non-clipped controller is used.

    Why inequality? Because non-clipped control requires interpolation around 't1',
    and filters and smoothers interpolate differently.
    """
    atol, rtol = 1e-2, 1e-1
    filter_solution = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=ivpsolvers.DynamicSolver(strategy=fil),
        atol=atol,
        rtol=rtol,
        control=ctrl,
    )
    smoother_solution = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=ivpsolvers.DynamicSolver(strategy=smo),
        atol=atol,
        rtol=rtol,
        control=ctrl,
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    # Equal:
    assert _tree_all_allclose(filter_solution.t, smoother_solution.t)

    filter_scale = filter_solution.output_scale
    smoother_scale = smoother_solution.output_scale
    assert _tree_all_allclose(filter_scale, smoother_scale)

    # Not-equal:
    assert not _tree_all_allclose(filter_solution.u, smoother_solution.u)
    filter_marginals = filter_solution.marginals
    smoother_marginals = smoother_solution.marginals
    assert not _tree_all_allclose(filter_marginals.mean, smoother_marginals.mean)

    filter_cov_sqrtm = filter_marginals.cov_sqrtm_lower
    smoother_cov_sqrtm = smoother_marginals.cov_sqrtm_lower
    assert not _tree_all_allclose(cov(filter_cov_sqrtm), cov(smoother_cov_sqrtm))


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
