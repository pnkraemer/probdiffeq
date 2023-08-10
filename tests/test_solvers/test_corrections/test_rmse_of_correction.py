"""Assert that every recipe yields a decent ODE approximation."""
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.statespace import correction, cubature, extrapolation
from probdiffeq.solvers.strategies import filters
from tests.setup import setup


@testing.case()
def case_ts0():
    try:
        return correction.taylor_order_zero()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_ts1():
    try:
        return correction.taylor_order_one()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr0():
    try:
        return correction.statistical_order_zero()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1():
    try:
        return correction.statistical_order_one()
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.case()
def case_slr1_gauss_hermite():
    try:
        return correction.statistical_order_one(cubature_fun=cubature.gauss_hermite)
    except NotImplementedError:
        return "not_implemented"
    raise RuntimeError


@testing.fixture(name="solution")
@testing.parametrize_with_cases("correction_impl", cases=".", prefix="case_")
def fixture_solution(correction_impl):
    vf, u0, (t0, t1) = setup.ode()

    problem_args = (vf, u0)
    problem_kwargs = {"t0": t0, "t1": t1}

    if correction_impl == "not_implemented":
        testing.skip(reason="This type of linearisation has not been implemented.")

    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    strategy = filters.filter_adaptive(ibm, correction_impl)
    solver = calibrated.mle(strategy)

    adaptive_kwargs = {
        "solver": solver,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_scale": jnp.ones_like(impl.ssm_util.prototype_output_scale()),
    }
    return ivpsolve.simulate_terminal_values(
        *problem_args, **problem_kwargs, **adaptive_kwargs
    )


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution():
    vf, (u0,), (t0, t1) = setup.ode()

    # Solve the IVP
    @jax.jit
    def vf_diffrax(t, y, args):
        return vf(y, t=t, p=args)

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
