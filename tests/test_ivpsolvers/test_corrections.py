"""Assert that every recipe yields a decent ODE approximation."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import functools, ode, testing
from probdiffeq.backend import numpy as np


@testing.case()
def case_ts0():
    return ivpsolvers.correction_ts0


@testing.case()
def case_ts1():
    return ivpsolvers.correction_ts1


@testing.case()
def case_slr0():
    return ivpsolvers.correction_slr0


@testing.case()
def case_slr1():
    return ivpsolvers.correction_slr1


@testing.case()
def case_slr1_gauss_hermite():
    return functools.partial(
        ivpsolvers.correction_slr1, cubature_fun=ivpsolvers.cubature_gauss_hermite
    )


@testing.fixture(name="solution")
@testing.parametrize_with_cases("correction_impl", cases=".", prefix="case_")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution(correction_impl, fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    try:
        tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
        init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
        corr = correction_impl(vf, ssm=ssm, damp=1e-9)

    except NotImplementedError:
        testing.skip(reason="This type of linearisation has not been implemented.")

    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=corr, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
    return ivpsolve.solve_adaptive_terminal_values(
        init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )


def test_terminal_value_simulation_matches_reference(solution):
    expected = reference_solution(solution.t)
    received = solution.u[0]
    assert np.allclose(received, expected, rtol=1e-2)


@functools.jit
def reference_solution(t1):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    ts = np.asarray([t0, t1])
    sol = ode.odeint_and_save_at(vf, (u0,), save_at=ts, atol=1e-10, rtol=1e-10)
    return sol[-1]
