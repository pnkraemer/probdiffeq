"""There are too many ways to smooth. We assert they all do the same."""

import jax.numpy as jnp
import pytest
import pytest_cases

from probdiffeq import solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers

# todo: both this file and test_checkpoint_same_grid.py call
#  solve_with_python_while_loop(... solver=smo) and solve_and_save_at(solver=fp_smo)
#  this redundancy should be eliminated


@pytest_cases.case
def smoother_pair_smoother_and_fixedpoint():
    impl = recipes.IsoTS0.from_params()
    return smoothers.Smoother(impl), smoothers.FixedPointSmoother(impl)


@pytest_cases.case
def smoother_pair_two_smoothers():
    # if the checkpoints are equal to the solver states,
    # then the checkpoint-simulator replicates _exactly_ what the non-checkpoint-
    # smoother does. So the tests must also pass in this setup.
    impl = recipes.IsoTS0.from_params()
    return smoothers.Smoother(impl), smoothers.Smoother(impl)


# Why a filter-warning?
#   We plug a non-fixed-point smoother into the checkpoint simulation
#   which does not work, UNLESS the smoother happens to step exactly
#   from checkpoint to checkpoint (which is the corner case that we are
#   testing here). Therefore, we happily ignore the warning.
@pytest.mark.filterwarnings("ignore:A conventional smoother")
@pytest_cases.parametrize_with_cases("smo, fp_smo", cases=".", prefix="smoother_pair_")
@pytest_cases.parametrize("tol", [1e-2])
@pytest_cases.parametrize_with_cases("ode_problem", cases="..problem_cases")
def test_smoothing_checkpoint_equals_solver_state(ode_problem, smo, fp_smo, tol):
    """In solve_and_save_at(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve_with_python_while_loop(), \
     the results should be identical."""
    smo_sol = solution_routines.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solvers.DynamicSolver(strategy=smo),
        atol=1e-2 * tol,
        rtol=tol,
    )

    fp_smo_sol = solution_routines.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=smo_sol.t,
        parameters=ode_problem.args,
        solver=solvers.DynamicSolver(strategy=fp_smo),
        atol=1e-2 * tol,
        rtol=tol,
    )

    tols = {"atol": 1e-2, "rtol": 1e-2}
    assert jnp.allclose(fp_smo_sol.t, smo_sol.t, **tols)
    assert jnp.allclose(fp_smo_sol.u, smo_sol.u, **tols)
    assert jnp.allclose(
        fp_smo_sol.marginals.hidden_state.mean,
        smo_sol.marginals.hidden_state.mean,
        **tols
    )
    assert jnp.allclose(
        fp_smo_sol.posterior.backward_model.noise.mean,
        smo_sol.posterior.backward_model.noise.mean,
        **tols
    )
    assert jnp.allclose(
        fp_smo_sol.output_scale_sqrtm, smo_sol.output_scale_sqrtm, **tols
    )

    # covariances are equal, but cov_sqrtm_lower might not be

    def cov(x):
        return jnp.einsum("...jk,...lk->...jl", x, x)

    l0 = fp_smo_sol.marginals.hidden_state.cov_sqrtm_lower
    l1 = smo_sol.marginals.hidden_state.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1), **tols)

    l0 = fp_smo_sol.posterior.backward_model.noise.cov_sqrtm_lower
    l1 = smo_sol.posterior.backward_model.noise.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1), **tols)
