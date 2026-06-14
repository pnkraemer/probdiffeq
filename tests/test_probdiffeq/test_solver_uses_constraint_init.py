"""Assert that the solver uses the constraint initialisation."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.fixture(name="ivp")
def ivp_lotka_volterra():
    """Return the Lotka-Volterra IVP tuple."""
    return ode.ivp_lotka_volterra()


def case_solver_standard():
    """Use the base solver."""
    return probdiffeq.solver


def case_solver_mle():
    """Use the MLE solver."""
    return probdiffeq.solver_mle


def case_solver_dynamic():
    """Use the dynamic solver."""
    return probdiffeq.solver_dynamic


@testing.parametrize("derivatives", [1, 4])
@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize_with_cases("solver_factory", cases=".", prefix="case_solver_")
def test_output_matches_reference(
    ivp, solver_factory, derivatives, ssm_factory
) -> None:
    """Assert that the solver uses the constraint init to set the first-step state accurately."""
    vf, (u0,), (t0, t1) = ivp

    @func.partial(probdiffeq.residual_jet_lift, lift_by=derivatives - 1)
    @probdiffeq.residual_velocity
    def residual(u, du, /, *, t):
        return tree.tree_map(
            lambda a, b: a + b,
            residual_linear(u, du, t=t),
            residual_nonlinear(u, du, t=t),
        )

    def residual_linear(_u, du, *, t):
        del t
        return du

    def residual_nonlinear(u, _du, *, t):
        vfu = vf(u, t=t)
        return tree.tree_map(lambda a: -a, vfu)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=derivatives)
    expected, _ = jetexpand(probdiffeq.ode(vf), [u0], t=t0)

    # Build an SSM (no ODE-jets, so that we can test the update at init)
    # Only use the dense factorisation because this test uses JET constraints
    # and they have not been implemented for isotropic or blockdiagonal models
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated([u0], diffuse_derivatives=derivatives)

    # Build a solver
    nlstsq = probdiffeq.lstsq_constrained_gauss_newton(maxiter=50, tol=1e-10)
    strategy = probdiffeq.strategy_filter()
    taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq)
    constraint = ssm.constraint_residual(residual, taylor_point=taylor_point)
    solver = solver_factory(
        strategy=strategy, constraint=constraint, constraint_init=constraint
    )
    solve = ivpsolve.solve_fixed_grid(solver=solver)

    # Compute the PN solution
    grid = np.linspace(t0, t1, endpoint=True, num=10)
    solution = func.jit(solve)(iwp, grid=grid)
    received = tree.tree_map(lambda s: s[0], solution.u.mean)
    assert testing.allclose(received, expected)
