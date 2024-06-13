"""Assert that every recipe yields a decent ODE approximation."""

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solvers, strategies
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.case()
def case_ts0():
    try:
        return components.ts0()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_ts1():
    try:
        return components.ts1()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr0():
    try:
        return components.slr0()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1():
    try:
        return components.slr1()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1_gauss_hermite():
    try:
        return components.slr1(cubature_fun=components.gauss_hermite)
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.fixture(name="solution")
@testing.parametrize_with_cases("correction_impl", cases=".", prefix="case_")
def fixture_solution(correction_impl):
    vf, u0, (t0, t1) = setup.ode()

    if correction_impl == "not_implemented":
        testing.skip(reason="This type of linearisation has not been implemented.")

    ibm = components.ibm_adaptive(num_derivatives=2)
    strategy = strategies.filter_adaptive(ibm, correction_impl)
    solver = solvers.mle(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    adaptive_kwargs = {"adaptive_solver": adaptive_solver, "dt0": 0.1}

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.simulate_terminal_values(vf, init, t0=t0, t1=t1, **adaptive_kwargs)


@testing.fixture(name="reference_solution")
def fixture_reference_solution():
    vf, (u0,), (t0, t1) = setup.ode()
    return ode.odeint_dense(vf, (u0,), t0=t0, t1=t1, atol=1e-10, rtol=1e-10)


def test_terminal_value_simulation_matches_reference(solution, reference_solution):
    expected = reference_solution(solution.t)
    received = solution.u

    assert np.allclose(received, expected, rtol=1e-2)
