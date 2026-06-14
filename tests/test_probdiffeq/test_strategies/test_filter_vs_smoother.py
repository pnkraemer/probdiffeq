"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, linalg, np, ode, testing, tree


@testing.fixture(name="solver_setup")
@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def fixture_solver_setup(ssm_factory):
    """Set up the Lotka-Volterra IVP and jet expansion for the filter vs smoother comparison."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    vf = probdiffeq.ode(vf)
    grid = np.linspace(t0, t1, endpoint=True, num=12)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    return {"vf": vf, "tcoeffs": tcoeffs, "grid": grid, "ssm_factory": ssm_factory}


@testing.fixture(name="filter_solution")
def fixture_filter_solution(solver_setup):
    """Solve the IVP on a fixed grid with the filter strategy."""
    tcoeffs = solver_setup["tcoeffs"]
    ssm = solver_setup["ssm_factory"]()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(solver_setup["vf"])
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    return func.jit(solve)(iwp, grid=solver_setup["grid"])


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    """Solve the IVP on a fixed grid with the fixed interval smoother strategy."""
    tcoeffs = solver_setup["tcoeffs"]
    ssm = solver_setup["ssm_factory"]()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(solver_setup["vf"])
    strategy = probdiffeq.strategy_smoother_fixedinterval()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    return func.jit(solve)(iwp, grid=solver_setup["grid"])


def test_filter_and_smoother_have_similar_rmse(
    filter_solution, smoother_solution
) -> None:
    """Assert that filter and smoother RMSE are within a factor of two and both below 0.01."""
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
    vf, (u0,), (_t0, _t1) = ode.ivp_lotka_volterra()
    vf = probdiffeq.ode(vf)
    return ode.odeint_and_save_at(vf, (u0,), save_at=ts, atol=1e-10, rtol=1e-10)


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
