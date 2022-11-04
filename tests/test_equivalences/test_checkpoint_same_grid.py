"""There are too many ways to smooth. We assert they all do the same."""

import jax.numpy as jnp
import pytest
import pytest_cases

from odefilter import ivpsolve, solvers
from odefilter.strategies import smoothers

# todo: both this file and test_checkpoint_same_grid.py call
#  solve(... solver=eks) and simulate_checkpoints(solver=fp_eks)


@pytest_cases.case
def smoother_pair_fixedpoint_eks0():
    return smoothers.Smoother(), smoothers.FixedPointSmoother()


@pytest_cases.case
def smoother_pair_two_eks0():
    # if the checkpoints are equal to the solver states,
    # then the checkpoint-simulator replicates _exactly_ what the non-checkpoint-
    # smoother does. So the tests must also pass in this setup.
    return smoothers.Smoother(), smoothers.Smoother()


# Why a filter-warning?
#   We plug a non-fixed-point smoother into the checkpoint simulation
#   which does not work, UNLESS the smoother happens to step exactly
#   from checkpoint to checkpoint (which is the corner case that we are
#   testing here). Therefore, we happily ignore the warning.
@pytest.mark.filterwarnings("ignore:A conventional smoother")
@pytest_cases.parametrize_with_cases("eks, fp_eks", cases=".", prefix="smoother_pair_")
@pytest_cases.parametrize("tol", [1e-2])
def test_smoothing_checkpoint_equals_solver_state(ode_problem, eks, fp_eks, tol):
    """In simulate_checkpoints(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve(), the results should be identical."""
    vf, u0, t0, t1, p = ode_problem
    eks_sol = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=eks),
        atol=1e-2 * tol,
        rtol=tol,
    )

    fp_eks_sol = ivpsolve.simulate_checkpoints(
        vf,
        u0,
        ts=eks_sol.t,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=fp_eks),
        atol=1e-2 * tol,
        rtol=tol,
    )

    tols = {"atol": 1e-2, "rtol": 1e-2}
    assert jnp.allclose(fp_eks_sol.t, eks_sol.t, **tols)
    assert jnp.allclose(fp_eks_sol.u, eks_sol.u, **tols)
    assert jnp.allclose(fp_eks_sol.marginals.mean, eks_sol.marginals.mean, **tols)
    assert jnp.allclose(
        fp_eks_sol.posterior.backward_model.noise.mean,
        eks_sol.posterior.backward_model.noise.mean,
        **tols
    )
    assert jnp.allclose(
        fp_eks_sol.output_scale_sqrtm, eks_sol.output_scale_sqrtm, **tols
    )

    # covariances are equal, but cov_sqrtm_lower might not be

    def cov(x):
        return jnp.einsum("...jk,...lk->...jl", x, x)

    l0 = fp_eks_sol.marginals.cov_sqrtm_lower
    l1 = eks_sol.marginals.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1), **tols)

    l0 = fp_eks_sol.posterior.backward_model.noise.cov_sqrtm_lower
    l1 = eks_sol.posterior.backward_model.noise.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1), **tols)
