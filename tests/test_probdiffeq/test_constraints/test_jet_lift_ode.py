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
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)

    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)

    # Compute a save-at solution
    ts = np.linspace(t0, t1 / 2.0, num=75, endpoint=True)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution = func.jit(solve)(iwp, grid=ts)

    print(solution.u.mean)
    assert False
