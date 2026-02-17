"""Assert that every recipe yields a decent ODE approximation."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, testing, tree


@testing.case()
def case_ts0():
    return probdiffeq.constraint_ode_ts0


@testing.case()
def case_ts1():
    return probdiffeq.constraint_ode_ts1


@testing.case()
def case_slr0():
    return probdiffeq.constraint_ode_slr0


@testing.case()
def case_slr1():
    return probdiffeq.constraint_ode_slr1


@testing.case()
def case_slr1_gauss_hermite():
    return func.partial(
        probdiffeq.constraint_ode_slr1, cubature_fun=probdiffeq.cubature_gauss_hermite
    )


@testing.fixture(name="solution")
@testing.parametrize_with_cases("constraint_ode_factory", cases=".", prefix="case_")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution(constraint_ode_factory, fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    try:
        tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
        constraint = constraint_ode_factory(vf, ssm=ssm)

    except NotImplementedError:
        reason = "Skipped test since NotImplementedError has been raised."
        reason += " Most likely because this combo of"
        reason += " linearisation + ssm factorisation isn't available."
        testing.skip(reason=reason)

    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_mle(
        strategy=strategy, prior=ibm, constraint=constraint, ssm=ssm
    )
    errorest = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)
    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, errorest=errorest)
    return solve(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2, damp=1e-9)


def test_terminal_value_simulation_matches_reference(solution):
    expected = reference_solution(solution.t)
    received = solution.u.mean[0]
    assert testing.allclose(received, expected, rtol=1e-2)


@func.jit
def reference_solution(t1):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    ts = np.asarray([t0, t1])
    sol = ode.odeint_and_save_at(vf, (u0,), save_at=ts, atol=1e-10, rtol=1e-10)
    return tree.tree_map(lambda s: s[-1], sol)
