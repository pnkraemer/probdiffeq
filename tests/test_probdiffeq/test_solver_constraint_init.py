"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, linalg, np, ode, testing, tree


@testing.fixture(name="ivp")
def ivp_lotka_volterra():
    return ode.ivp_lotka_volterra()


def case_solver_standard():
    return probdiffeq.solver


def case_solver_mle():
    return probdiffeq.solver_mle


def case_solver_dynamic():
    return probdiffeq.solver_dynamic


@testing.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
def test_output_matches_reference(ivp, solver) -> None:
    vf, (u0,), (t0, t1) = ivp

    def root(u, du, *, t):
        return tree.tree_map(lambda a, b: a - b, du, vf(u, t=t))

    # Build an SSM (no ODE-jets, so that we can test the update at init)
    # Only use the dense factorisation because this test uses JET constraints
    # and they have not been implemented for isotropic or blockdiagonal models
    init, ssm = probdiffeq.ssm_taylor([u0], diffuse_derivatives=4)

    # Build a solver
    prior = probdiffeq.prior_wiener_integrated(ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    constraint = probdiffeq.constraint_root_jet(root, ssm=ssm)
    solver = solver(
        strategy=strategy,
        prior=prior,
        constraint=constraint,
        ssm=ssm,
        constraint_init=constraint,
    )
    error = probdiffeq.error_state_std(prior=prior, ssm=ssm, constraint=constraint)

    # Compute the PN solution
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    received = func.jit(solve)(init, save_at=save_at, atol=1e-4, rtol=1e-4)

    # Compute a reference solution
    expected = ode.odeint_and_save_at(vf, (u0,), save_at=save_at, atol=1e-7, rtol=1e-7)

    # The results should be very similar
    assert testing.allclose(received.u.mean[0], expected)

    # The posterior standard deviation at the initial time-point
    # should essentially be zero if the initial constraint did something
    stds = tree.tree_leaves(received.u.std)
    tol = np.sqrt(np.finfo_eps(stds[-1][0].dtype))
    assert linalg.vector_norm(stds[-1][0]) < tol
