from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
def test_jet_lift_ode_works(ssm) -> None:
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    ts = np.linspace(t0, t1 / 2.0, num=75, endpoint=True)
    strategy = probdiffeq.strategy_filter()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    # Compute a reference solution
    ts0 = ssm.constraint_ode_ts0(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_ts0 = func.jit(solve)(iwp, grid=ts)

    # Compute a jet-lifted solution
    ts0 = ssm.constraint_ode_ts0(vf.jet_lift(lift_by=2))
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_jet_lift = func.jit(solve)(iwp, grid=ts)

    print(solution_jet_lift.u.mean)
    assert testing.allclose(solution_ts0.u.mean, solution_jet_lift.u.mean)
    assert False
