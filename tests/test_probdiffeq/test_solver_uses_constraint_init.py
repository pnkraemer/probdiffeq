"""Assert that the base adaptive solver is accurate."""

from probdiffeq import diffeqjet, ivpsolve, probdiffeq
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

    def root(u, du, *, t):
        return tree.tree_map(
            lambda a, b: a + b, root_linear(u, du, t=t), root_nonlinear(u, du, t=t)
        )

    def root_linear(_u, du, *, t):
        del t
        return du

    def root_nonlinear(u, _du, *, t):
        vfu = vf(u, t=t)
        return tree.tree_map(lambda a: -a, vfu)

    expected = diffeqjet.odejet_padded_scan(
        lambda y: vf(y, t=t0), [u0], num=derivatives
    )

    # Build an SSM (no ODE-jets, so that we can test the update at init)
    # Only use the dense factorisation because this test uses JET constraints
    # and they have not been implemented for isotropic or blockdiagonal models
    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)
    init, prior = probdiffeq.prior_wiener_integrated(
        [u0], diffuse_derivatives=derivatives, ssm=ssm
    )

    # Build a solver
    nlstsq = probdiffeq.nlstsq_gauss_newton_weighted_constrained(maxiter=50, tol=1e-10)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    linearization = probdiffeq.linearization_map(nlstsq)
    constraint = probdiffeq.constraint(root, ssm=ssm, linearization=linearization)
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
