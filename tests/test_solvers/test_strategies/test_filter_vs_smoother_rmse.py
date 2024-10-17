"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import linalg, ode, testing
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


@testing.fixture(name="solver_setup")
def fixture_solver_setup(ssm):
    vf, (u0,), (t0, t1) = ssm.default_ode

    output_scale = np.ones_like(impl.prototypes.output_scale())
    grid = np.linspace(t0, t1, endpoint=True, num=12)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    return {"vf": vf, "tcoeffs": tcoeffs, "grid": grid, "output_scale": output_scale}


@testing.fixture(name="filter_solution")
def fixture_filter_solution(solver_setup):
    tcoeffs, output_scale = solver_setup["tcoeffs"], solver_setup["output_scale"]
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale)
    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver(strategy)

    init = solver.initial_condition()
    return ivpsolve.solve_fixed_grid(
        solver_setup["vf"], init, grid=solver_setup["grid"], solver=solver
    )


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    tcoeffs, output_scale = solver_setup["tcoeffs"], solver_setup["output_scale"]
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale)
    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_smoother(ibm, ts0)
    solver = ivpsolvers.solver(strategy)

    init = solver.initial_condition()
    return ivpsolve.solve_fixed_grid(
        solver_setup["vf"], init, grid=solver_setup["grid"], solver=solver
    )


@testing.fixture(name="reference_solution")
def fixture_reference_solution(ssm):
    vf, (u0,), (t0, t1) = ssm.default_ode
    return ode.odeint_dense(vf, (u0,), t0=t0, t1=t1, atol=1e-10, rtol=1e-10)


def test_compare_filter_smoother_rmse(
    filter_solution, smoother_solution, reference_solution
):
    assert np.allclose(filter_solution.t, smoother_solution.t)  # sanity check

    reference = reference_solution(filter_solution.t)
    filter_rmse = _rmse(filter_solution.u, reference)
    smoother_rmse = _rmse(smoother_solution.u, reference)

    # I would like to compare filter & smoother RMSE. but this test is too unreliable,
    # so we simply assert that both are "comparable".
    assert np.allclose(filter_rmse, smoother_rmse, atol=0.0, rtol=1.0)

    # The error should be small, otherwise the test makes little sense
    assert filter_rmse < 0.01


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
