"""Assert that every recipe yields a decent ODE approximation."""
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, cubature, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.case()
def case_ts0():
    try:
        return corrections.ts0()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_ts1():
    try:
        return corrections.ts1()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr0():
    try:
        return corrections.slr0()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1():
    try:
        return corrections.slr1()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1_gauss_hermite():
    try:
        return corrections.slr1(cubature_fun=cubature.gauss_hermite)
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.fixture(name="solution")
@testing.parametrize_with_cases("correction_impl", cases=".", prefix="case_")
def fixture_solution(correction_impl):
    vf, u0, (t0, t1) = setup.ode()

    if correction_impl == "not_implemented":
        testing.skip(reason="This type of linearisation has not been implemented.")

    ibm = priors.ibm_adaptive(num_derivatives=2)
    strategy = filters.filter_adaptive(ibm, correction_impl)
    solver = calibrated.mle(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    adaptive_kwargs = {"adaptive_solver": adaptive_solver, "dt0": 0.1}

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.simulate_terminal_values(vf, init, t0=t0, t1=t1, **adaptive_kwargs)


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution():
    vf, (u0,), (t0, t1) = setup.ode()

    # Solve the IVP
    @jax.jit
    def vf_diffrax(t, y, args):  # noqa: ARG001
        return vf(y, t=t)

    term = diffrax.ODETerm(vf_diffrax)
    solver = diffrax.Dopri5()
    solution_object = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=u0,
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )

    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_terminal_value_simulation_matches_diffrax(solution, diffrax_solution):
    expected = diffrax_solution(solution.t)
    received = solution.u

    assert jnp.allclose(received, expected, rtol=1e-2)
