"""Assert that solve_with_python_loop is accurate."""
import diffeqzoo.ivps
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # noqa: ARG001
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.fixture(name="python_loop_solution")
def fixture_python_loop_solution(problem):
    vf, u0, (t0, t1), f_args = problem

    problem_args = (vf, (u0,))
    problem_kwargs = {"t0": t0, "t1": t1, "parameters": f_args}

    solver = test_util.generate_solver(num_derivatives=4, ode_shape=(2,))
    adaptive_kwargs = {
        "solver": solver,
        "output_scale": 1.0,
        "atol": 1e-2,
        "rtol": 1e-2,
    }
    return ivpsolve.solve_with_python_while_loop(
        *problem_args, **problem_kwargs, **adaptive_kwargs
    )


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution(problem):
    vf, u0, (t0, t1), f_args = problem

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
        args=f_args,
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
