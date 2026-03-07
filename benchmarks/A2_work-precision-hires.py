"""Work vs precision - Hires."""

import functools
import statistics
import timeit
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq, taylor

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(start=1.0, stop=10.0, step=1.0, repeats=3) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

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
    tolerances = setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {
        r"ProbDiffEq: TS1($3$)": solver_probdiffeq(num_derivatives=3),
        r"ProbDiffEq: TS1($5$)": solver_probdiffeq(num_derivatives=5),
        r"ProbDiffEq: TS1($7$)": solver_probdiffeq(num_derivatives=7),
        "SciPy: 'LSODA'": solver_scipy(method="LSODA"),
        "SciPy: 'Radau'": solver_scipy(method="Radau"),
    }

    # Compute a reference solution
    reference = solver_scipy(method="BDF")(1e-13)
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    _fig, ax = plt.subplots(figsize=(5, 3))
    for label, wp in results.items():
        ax.loglog(wp["precision"], wp["work_mean"], label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small")

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


def setup_tolerances(*, start: float, stop: float, step: float) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(start, stop, step=step)


def setup_timeit(*, repeats: int) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=repeats))

    return timer


def solver_probdiffeq(*, num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @jax.jit
    def vf_probdiffeq(u, /, *, t):  # noqa: ARG001
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

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        vf_auto = functools.partial(vf_probdiffeq, t=t0)
        tcoeffs = taylor.odejet_padded_scan(vf_auto, (u0,), num=num_derivatives)
        init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")

        iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)
        ts1 = probdiffeq.constraint_ode_ts1(vf_probdiffeq, ssm=ssm)

        strategy = probdiffeq.strategy_filter(ssm=ssm)
        solver = probdiffeq.solver_dynamic(
            strategy=strategy, prior=iwp, constraint=ts1, ssm=ssm
        )
        error = probdiffeq.error_residual_std(constraint=ts1, prior=iwp, ssm=ssm)

        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)

        # Solve
        dt0 = ivpsolve.dt0(vf_auto, (u0,))

        solution = solve(init, t0=t0, t1=t1, dt0=dt0, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str) -> Callable:
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
        return jnp.asarray(solution.y[:, -1])

    return param_to_solution


def rmse_relative(expected: jax.Array, *, nugget=1e-5) -> Callable:
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        error_relative = error_absolute / jnp.abs(nugget + expected)
        return jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

    return rmse


def workprec(fun, *, precision_fun: Callable, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function into parameter-to-workprecision."""

    def parameter_list_to_workprecision(list_of_args, /):
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg).block_until_ready())
            times = timeit_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

            precisions.append(precision)
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


main()
