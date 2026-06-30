"""Tests for filter interpolation."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
def test_filter_marginals_accessible_in_smoother(ssm) -> None:
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    error = probdiffeq.error_residual_std(constraint=ts0)
    ts = np.linspace(t0, t1, num=15, endpoint=True)

    # Compute a filter solution
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution_filter = func.jit(solve)(iwp, atol=1e-1, rtol=1e-1, save_at=ts)

    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution_smoother = func.jit(solve)(iwp, atol=1e-1, rtol=1e-1, save_at=ts)

    (m1, C1) = solution_filter.u.to_multivariate_normal()
    (m2, C2) = solution_smoother.solution_full.marginal.to_multivariate_normal()

    assert testing.allclose(m1, m2)
    assert testing.allclose(C1, C2)
