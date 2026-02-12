"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import functools, ode, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.case
def case_strategy_filter():
    return probdiffeq.strategy_filter


@testing.case
def case_strategy_fixedpoint():
    return probdiffeq.strategy_fixedpoint


@testing.case
def case_solver_solver():
    return probdiffeq.solver


@testing.case
def case_solver_mle():
    return probdiffeq.solver_mle


@testing.case
def case_solver_dynamic():
    return probdiffeq.solver_dynamic


@testing.case
def case_correction_ts0():
    return probdiffeq.correction_ts0


@testing.case
def case_correction_ts1():
    return probdiffeq.correction_ts1


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
@testing.parametrize_with_cases("strategy_factory", ".", prefix="case_strategy_")
@testing.parametrize_with_cases("solver_factory", ".", prefix="case_solver_")
@testing.parametrize_with_cases("correction_factory", ".", prefix="case_correction_")
def test_output_matches_reference(
    fact, solver_factory, correction_factory, strategy_factory
):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Build a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    strat = strategy_factory(ssm=ssm)
    corr = correction_factory(vf, ssm=ssm)
    solver = solver_factory(strategy=strat, prior=iwp, correction=corr, ssm=ssm)
    errorest = probdiffeq.errorest_schober_bosch(prior=iwp, ssm=ssm, correction=corr)

    # Compute the PN solution
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
    received = functools.jit(solve)(init, save_at=save_at, atol=1e-3, rtol=1e-3)

    # Compute a reference solution
    expected = ode.odeint_and_save_at(vf, u0, save_at=save_at, atol=1e-7, rtol=1e-7)

    # The results should be very similar
    assert testing.allclose(received.u.mean[0], expected)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, received.u.mean)
    u_std_shape = tree_util.tree_map(np.shape, received.u.std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
