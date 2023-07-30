"""Compare solve_fixed_grid to solve_with_python_while_loop."""


import diffeqzoo.ivps
import jax

from probdiffeq import controls, ivpsolve, test_util
from probdiffeq.backend import testing


def test_fixed_grid_result_matches_adaptive_grid_result():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    problem_args = (vf, (u0,))
    problem_kwargs = {"parameters": f_args}

    solver = test_util.generate_solver(num_derivatives=2)
    adaptive_kwargs = {
        "t0": t0,
        "t1": t1,
        "solver": solver,
        "atol": 1e-2,
        "rtol": 1e-2,
        # Any clipped controller will do.
        "control": controls.IntegralClipped(),
    }
    solution_adaptive = ivpsolve.solve_with_python_while_loop(
        *problem_args, **problem_kwargs, **adaptive_kwargs
    )

    grid_adaptive = solution_adaptive.t
    solution_fixed = ivpsolve.solve_fixed_grid(
        *problem_args, grid=grid_adaptive, parameters=f_args, solver=solver
    )
    assert testing.tree_all_allclose(solution_adaptive, solution_fixed)
