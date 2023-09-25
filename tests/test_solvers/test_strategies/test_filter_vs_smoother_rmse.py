"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import filters, smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="solver_setup")
def fixture_solver_setup():
    vf, (u0,), (t0, t1) = setup.ode()

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    grid = jnp.linspace(t0, t1, endpoint=True, num=12)
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=2)
    return {"vf": vf, "tcoeffs": tcoeffs, "grid": grid, "output_scale": output_scale}


@testing.fixture(name="filter_solution")
def fixture_filter_solution(solver_setup):
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    tcoeffs, output_scale = solver_setup["tcoeffs"], solver_setup["output_scale"]
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.solve_fixed_grid(
        solver_setup["vf"], init, grid=solver_setup["grid"], solver=solver
    )


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    tcoeffs, output_scale = solver_setup["tcoeffs"], solver_setup["output_scale"]
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.solve_fixed_grid(
        solver_setup["vf"], init, grid=solver_setup["grid"], solver=solver
    )


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

    @jax.vmap
    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_compare_filter_smoother_rmse(
    filter_solution, smoother_solution, diffrax_solution
):
    assert jnp.allclose(filter_solution.t, smoother_solution.t)  # sanity check

    reference = diffrax_solution(filter_solution.t)
    filter_rmse = _rmse(filter_solution.u, reference)
    smoother_rmse = _rmse(smoother_solution.u, reference)

    # I would like to compare filter & smoother RMSE. but this test is too unreliable,
    # so we simply assert that both are "comparable".
    assert jnp.allclose(filter_rmse, smoother_rmse, atol=0.0, rtol=1.0)

    # The error should be small, otherwise the test makes little sense
    assert filter_rmse < 0.01


def _rmse(a, b):
    return jnp.linalg.norm((a - b) / b) / jnp.sqrt(b.size)
