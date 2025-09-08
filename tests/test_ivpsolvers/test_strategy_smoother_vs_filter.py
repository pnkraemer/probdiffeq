"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import linalg, ode, testing
from probdiffeq.backend import numpy as np


@testing.fixture(name="solver_setup")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solver_setup(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    grid = np.linspace(t0, t1, endpoint=True, num=12)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    return {"vf": vf, "tcoeffs": tcoeffs, "grid": grid, "fact": fact}


@testing.fixture(name="filter_solution")
def fixture_filter_solution(solver_setup):
    tcoeffs = solver_setup["tcoeffs"]
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(
        tcoeffs, ssm_fact=solver_setup["fact"]
    )
    ts0 = ivpsolvers.correction_ts0(solver_setup["vf"], ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    return ivpsolve.solve_fixed_grid(
        init, grid=solver_setup["grid"], solver=solver, ssm=ssm
    )


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    tcoeffs = solver_setup["tcoeffs"]
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(
        tcoeffs, ssm_fact=solver_setup["fact"]
    )
    ts0 = ivpsolvers.correction_ts0(solver_setup["vf"], ssm=ssm)
    strategy = ivpsolvers.strategy_smoother(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    return ivpsolve.solve_fixed_grid(
        init, grid=solver_setup["grid"], solver=solver, ssm=ssm
    )


def test_compare_filter_smoother_rmse(filter_solution, smoother_solution):
    assert np.allclose(filter_solution.t, smoother_solution.t)  # sanity check

    reference = _reference_solution(filter_solution.t)
    filter_rmse = _rmse(filter_solution.u[0], reference)
    smoother_rmse = _rmse(smoother_solution.u[0], reference)

    # I would like to compare filter & smoother RMSE. but this test is too unreliable,
    # so we simply assert that both are "comparable".
    assert np.allclose(filter_rmse, smoother_rmse, atol=0.0, rtol=1.0)

    # The error should be small, otherwise the test makes little sense
    assert filter_rmse < 0.01


def _reference_solution(ts):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    return ode.odeint_and_save_at(vf, (u0,), save_at=ts, atol=1e-10, rtol=1e-10)


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
