"""Walltime | Hires."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# Stiffness + high accuracy requires f64
jax.config.update("jax_enable_x64", True)


def main(start=3.0, stop=10.0, step=1.0, repeats=1) -> None:
    """Run the script."""
    # Simulate once to plot the state
    ts, ys = solve_ivp_once()

    _fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ts, ys)
    ax.set_title("Hires problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = 0.1 ** jnp.arange(start, stop, step=step)

    # Compute a reference solution
    reference = solver_scipy(method="BDF", precision_fun=lambda x: x)(1e-13)
    precision_fun = benchmark_util.rmse_relative(reference)

    # Assemble algorithms
    algorithms = {
        r"TS1($3$, matfree)": solver_matfree(
            num_derivatives=3, precision_fun=precision_fun
        ),
        r"TS1($5$, matfree)": solver_matfree(
            num_derivatives=5, precision_fun=precision_fun
        ),
        r"TS1($5$, dense)": solver_dense(
            num_derivatives=5, precision_fun=precision_fun
        ),
        r"TS1($7$, dense)": solver_dense(
            num_derivatives=7, precision_fun=precision_fun
        ),
        "SciPy: 'LSODA'": solver_scipy(method="LSODA", precision_fun=precision_fun),
        "SciPy: 'Radau'": solver_scipy(method="Radau", precision_fun=precision_fun),
    }

    # Compute all work-precision diagrams
    _fig, ax = plt.subplots(figsize=(8, 4))
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=repeats)
        wp = param_to_wp(tolerances)
        ax.loglog(wp.precision, wp.work.mean(axis=-1), ".-", label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def solve_ivp_once():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, u):
        """High irradiance response."""
        du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
        du2 = 1.71 * u[0] - 8.75 * u[1]
        du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
        du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
        du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
        du6 = (
            -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
        )
        du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
        du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
        return np.asarray([du1, du2, du3, du4, du5, du6, du7, du8])

    u0 = np.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
    time_span = np.asarray([0.0, 321.8122])

    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def solver_matfree(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf(u, /, *, t):  # noqa: ARG001
        """High irradiance response."""
        du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
        du2 = 1.71 * u[0] - 8.75 * u[1]
        du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
        du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
        du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
        du6 = (
            -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
        )
        du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
        du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
        return jnp.asarray([du1, du2, du3, du4, du5, du6, du7, du8])

    u0 = jnp.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
    t0, t1 = jnp.asarray([0.0, 321.8122])

    key = jax.random.PRNGKey(1)
    num_ensembles = (num_derivatives + 1) * 2
    ssm = probdiffeq.state_space_model_matfree(key=key, num_ensembles=num_ensembles)

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts1 = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ts1)
        error = probdiffeq.error_state_std(constraint=ts1)
        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)

        # Solve
        dt0 = ivpsolve.dt0(vf, (u0,), t=t0)
        solution = solve(iwp, t0=t0, t1=t1, dt0=dt0, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_dense(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf(u, /, *, t):  # noqa: ARG001
        """High irradiance response."""
        du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
        du2 = 1.71 * u[0] - 8.75 * u[1]
        du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
        du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
        du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
        du6 = (
            -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
        )
        du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
        du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
        return jnp.asarray([du1, du2, du3, du4, du5, du6, du7, du8])

    u0 = jnp.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
    t0, t1 = jnp.asarray([0.0, 321.8122])
    ssm = probdiffeq.state_space_model_dense()

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts1 = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ts1)
        error = probdiffeq.error_residual_std(constraint=ts1)
        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)

        # Solve
        dt0 = ivpsolve.dt0(vf, (u0,), t=t0)
        solution = solve(iwp, t0=t0, t1=t1, dt0=dt0, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, precision_fun) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, u):
        """High irradiance response."""
        du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
        du2 = 1.71 * u[0] - 8.75 * u[1]
        du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
        du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
        du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
        du6 = (
            -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
        )
        du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
        du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
        return np.asarray([du1, du2, du3, du4, du5, du6, du7, du8])

    u0 = np.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
    time_span = np.asarray([0.0, 321.8122])

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            vf_scipy,
            y0=u0,
            t_span=time_span,
            t_eval=time_span,
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return precision_fun(solution.y[:, -1])

    return param_to_solution


main()
