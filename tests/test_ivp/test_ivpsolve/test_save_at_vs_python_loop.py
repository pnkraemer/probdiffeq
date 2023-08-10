"""Assert that solve_and_save_at is consistent with solve_with_python_loop()."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.backend import testing


def test_save_at_result_matches_interpolated_adaptive_result():
    """Test that the save_at result matches the interpolation (using a filter)."""
    # Make a problem
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # noqa: ARG001
        return f(x, *p)

    problem_args = (vf, (u0,))
    problem_kwargs = {"parameters": f_args}

    # Generate a solver
    solver = test_util.generate_solver(num_derivatives=2, ode_shape=(2,))
    adaptive_kwargs = {
        "solver": solver,
        "output_scale": 1.0,
        "atol": 1e-2,
        "rtol": 1e-2,
    }

    # Compute an adaptive solution and interpolate
    ts = jnp.linspace(t0, t1, num=15, endpoint=True)
    solution_adaptive = ivpsolve.solve_with_python_while_loop(
        *problem_args, **problem_kwargs, t0=t0, t1=t1, **adaptive_kwargs
    )
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_adaptive, solver=solver
    )

    # Compute a save-at solution and remove the edge-points
    solution_save_at = ivpsolve.solve_and_save_at(
        *problem_args, **problem_kwargs, save_at=ts, **adaptive_kwargs
    )

    u_save_at = solution_save_at.u[1:-1]
    marginals_save_at = jax.tree_util.tree_map(
        lambda s: s[1:-1], solution_save_at.marginals
    )

    # Assert similarity
    assert jnp.allclose(u_interp, u_save_at)
    assert testing.marginals_allclose(marginals_interp, marginals_save_at)
