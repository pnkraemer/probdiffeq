"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve_with_python_while_loop() calls with default smoothers.
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.ssm import recipes
from probdiffeq.strategies import filters, smoothers


@testing.case
def strategy_pair_smoother():
    impl = recipes.ts0_iso()
    return filters.Filter(impl), smoothers.Smoother(impl)


@testing.case
def strategy_pair_fixedpoint_smoother():
    impl = recipes.ts0_iso()
    return filters.Filter(impl), smoothers.FixedPointSmoother(impl)


@testing.parametrize_with_cases("fil, smo", cases=".", prefix="strategy_pair_")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
def test_final_state_equal_to_filter(ode_problem, fil, smo):
    """Filters and smoothers should compute the same terminal values."""
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
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    assert _tree_all_allclose(filter_solution.t, smoother_solution.t)
    assert _tree_all_allclose(filter_solution.u, smoother_solution.u)
    assert _tree_all_allclose(
        filter_solution.marginals.hidden_state.mean,
        smoother_solution.marginals.hidden_state.mean,
    )
    assert _tree_all_allclose(
        cov(filter_solution.marginals.hidden_state.cov_sqrtm_lower),
        cov(smoother_solution.marginals.hidden_state.cov_sqrtm_lower),
    )
    assert _tree_all_allclose(
        filter_solution.output_scale, smoother_solution.output_scale
    )


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
