from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
@testing.parametrize("lift_by", [0, 1, 2])  # max: num_derivatives - 1
@testing.parametrize("num_derivatives", [3])
def test_jet_lift_ode_works(ssm, lift_by, num_derivatives) -> None:
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    ts = np.linspace(t0, t1, num=50, endpoint=True)
    strategy = probdiffeq.strategy_filter()

    # Build an ODE and a jet-lifted ODE
    # TODO: assert that jet_lift has a good error message if lift_by is too large
    vf = probdiffeq.ode(vf)
    vf_lifted = vf.jet_lift(lift_by=lift_by)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    # Compute a reference solution
    ts0 = ssm.constraint_ode_ts0(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_ts0 = func.jit(solve)(iwp, grid=ts)

    # Compute a jet-lifted solution
    ts0 = ssm.constraint_ode_ts0(vf_lifted)
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_jet_lift = func.jit(solve)(iwp, grid=ts)

    assert testing.allclose(
        solution_ts0.u.mean[0], solution_jet_lift.u.mean[0], atol=1e-3, rtol=1e-3
    )
