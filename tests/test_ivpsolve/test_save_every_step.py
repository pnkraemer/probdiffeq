"""Assert that solve_with_python_loop is accurate."""
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, timestep
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import adaptive, correction, extrapolation
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="python_loop_solution")
def fixture_python_loop_solution():
    vf, u0, (t0, t1) = setup.ode()

    ibm = extrapolation.ibm_adaptive(num_derivatives=4)
    ts0 = correction.taylor_order_zero()
    strategy = adaptive.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "solver": solver,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }
    return ivpsolve.solve_and_save_every_step(*args, **kwargs)


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution():
    vf, (u0,), (t0, t1) = setup.ode()

    # Solve the IVP
    @jax.jit
    def vf_diffrax(t, y, _args):
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

    @jax.vmap
    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_python_loop_output_matches_diffrax(python_loop_solution, diffrax_solution):
    expected = diffrax_solution(python_loop_solution.t)
    received = python_loop_solution.u

    assert jnp.allclose(received, expected, rtol=1e-3)
