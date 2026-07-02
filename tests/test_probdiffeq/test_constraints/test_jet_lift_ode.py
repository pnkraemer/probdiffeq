"""Tests for jet-lifting functionality for ODEs."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing

# TODO: make all existing tests pass, then add cases here to cover the remainders.
# (Eg higher order ODEs?)


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.case
def case_ssm_blockdiag():
    """Construct a block-diagonal SSM."""
    return probdiffeq.state_space_model_blockdiag()


@testing.case
def case_ssm_isotropic():
    """Construct an isotropic SSM."""
    return probdiffeq.state_space_model_isotropic()


@testing.case
def case_constraint_ts0():
    """Construct a TS0 constraint."""
    return lambda ssm, ode: ssm.constraint_ode_ts0(ode)


@testing.case
def case_constraint_ts1():
    """Construct a TS0 constraint."""
    return lambda ssm, ode: ssm.constraint_ode_ts1(ode)


@testing.case
def case_ode_lotka_volterra():
    """Construct a Lotka-Volterra ODE."""
    return ode.ivp_lotka_volterra()


@testing.parametrize("lift_by", [0, 1, 2])  # max: num_derivatives - 1
@testing.parametrize("num_derivatives", [3])
@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
@testing.parametrize_with_cases("constraint", cases=".", prefix="case_constraint_")
@testing.parametrize_with_cases("ode", cases=".", prefix="case_ode_")
def test_jet_lift_ode_does_not_reduce_accuracy(
    ssm, constraint, ode, lift_by, num_derivatives
):
    """Test that jet-lifting an ODE does not reduce accuracy."""
    vf, u0, (t0, t1) = ode

    # Generate a solver
    ts = np.linspace(t0, t1, num=50, endpoint=True)
    strategy = probdiffeq.strategy_filter()

    # Build an ODE and a jet-lifted ODE
    # TODO: assert that jet_lift has a good error message if lift_by is too large

    # Materialize the Jacobian to ensure that trace-estimation errors
    # are not the culprits for failing tests
    vf = probdiffeq.ode(vf, jacobian=probdiffeq.jacobian_materialize())
    vf_lifted = vf.jet_lift(lift_by=lift_by)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    # Compute a reference solution
    ts0 = constraint(ssm, vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_ts0 = func.jit(solve)(iwp, grid=ts)

    # Compute a jet-lifted solution
    ts0 = constraint(ssm, vf_lifted)
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_jet_lift = func.jit(solve)(iwp, grid=ts)

    assert testing.allclose(
        solution_ts0.u.mean[0], solution_jet_lift.u.mean[0], atol=1e-3, rtol=1e-3
    )
