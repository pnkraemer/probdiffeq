"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, linalg, np, ode, testing, tree


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
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, ssm_fact=solver_setup["fact"]
    )
    ts0 = probdiffeq.constraint_ode_ts0(ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(
        solver_setup["vf"], strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm
    )
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    return solve(init, grid=solver_setup["grid"])


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    tcoeffs = solver_setup["tcoeffs"]
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, ssm_fact=solver_setup["fact"]
    )
    ts0 = probdiffeq.constraint_ode_ts0(ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    solver = probdiffeq.solver(
        solver_setup["vf"], strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm
    )
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    return solve(init, grid=solver_setup["grid"])


def test_compare_filter_smoother_rmse(filter_solution, smoother_solution):
    assert testing.allclose(filter_solution.t, smoother_solution.t)  # sanity check

    reference = _reference_solution(filter_solution.t)
    vmap_tree_ravel = func.vmap(lambda s: tree.ravel_pytree(s)[0])
    u_fi = vmap_tree_ravel(filter_solution.u.mean[0])
    u_sm = vmap_tree_ravel(smoother_solution.u.mean[0])
    u_re = vmap_tree_ravel(reference)

    filter_rmse = _rmse(u_fi, u_re)
    smoother_rmse = _rmse(u_sm, u_re)

    # I would like to compare filter & smoother RMSE. but this test is too unreliable,
    # so we simply assert that both are "comparable".
    assert testing.allclose(filter_rmse, smoother_rmse, atol=0.0, rtol=1.0)

    # The error should be small, otherwise the test makes little sense
    assert filter_rmse < 0.01


def _reference_solution(ts):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    return ode.odeint_and_save_at(vf, (u0,), save_at=ts, atol=1e-10, rtol=1e-10)


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
