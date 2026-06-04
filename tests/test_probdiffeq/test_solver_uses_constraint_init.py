"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.fixture(name="ivp")
def ivp_lotka_volterra():
    return ode.ivp_lotka_volterra()


def case_solver_standard():
    return probdiffeq.solver


def case_solver_mle():
    return probdiffeq.solver_mle


def case_solver_dynamic():
    return probdiffeq.solver_dynamic


@testing.parametrize("derivatives", [1, 4])
@testing.parametrize("ssm_fact", ["dense"])
@testing.parametrize_with_cases("solver_factory", cases=".", prefix="case_solver_")
def test_output_matches_reference(ivp, solver_factory, derivatives, ssm_fact) -> None:
    vf, (u0,), (t0, t1) = ivp

    @func.partial(probdiffeq.jet_lift, lift_by=derivatives - 1)
    @probdiffeq.residual_state_velocity
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
    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)
    init, prior = probdiffeq.prior_wiener_integrated(
        [u0], diffuse_derivatives=derivatives, ssm=ssm
    )

    # Build a solver
    nlstsq = probdiffeq.lstsq_constrained_gauss_newton(maxiter=50, tol=1e-10)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq)
    constraint = probdiffeq.constraint_residual(
        residual, ssm=ssm, taylor_point=taylor_point
    )
    solver = solver_factory(
        strategy=strategy,
        prior=prior,
        constraint=constraint,
        ssm=ssm,
        constraint_init=constraint,
    )
    solve = ivpsolve.solve_fixed_grid(solver=solver)

    # Compute the PN solution
    grid = np.linspace(t0, t1, endpoint=True, num=10)
    solution = func.jit(solve)(init, grid=grid)
    received = tree.tree_map(lambda s: s[0], solution.u.mean)
    assert testing.allclose(received, expected)
