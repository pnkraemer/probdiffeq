"""Convergence rates | Lotka-Volterra."""

import statistics
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main() -> None:
    """Run the script."""
    # High order solvers need double precision
    jax.config.update("jax_enable_x64", True)

    # Set up the benchmark (compute a reference etc.)
    reference = solver_scipy(method="LSODA")(1e-12)
    tolerances = 0.1 ** jnp.arange(2, 8, step=0.5)
    precision_fun = benchmark_util.rmse_relative(reference)

    # Assemble algorithms
    algorithms = {}
    for n in range(1, 17):
        algorithms[f"TS1({n})"] = solver_probdiffeq(n, precision_fun=precision_fun)

    # Compute all work-precision diagrams
    layout = [["values", "trends"]]
    _fig, ax = plt.subplot_mosaic(
        layout,
        figsize=(8, 3),
        constrained_layout=True,
        dpi=120,
        sharex=True,
        sharey=True,
    )
    pbar = tqdm.tqdm(algorithms.items())
    for i, (label, algo) in enumerate(pbar):
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=0)
        wp = param_to_wp(tolerances)

        cmap = mpl.colormaps["managua"]
        i_clipped = i / len(algorithms.keys())
        color = mpl.colors.to_hex(cmap(i_clipped))

        # Compute linear trend
        x, y = wp.precision["num_steps"], wp.precision["precision"]
        (x_lin, y_lin), (scale, _) = linear_trend(x, y)

        # All curves start at (1, 1)
        ax["values"].loglog(x / x.min(), y / y.max(), ".-", color=color, label=label)
        ax["trends"].loglog(
            x_lin / x_lin.min(),
            y_lin / y_lin.max(),
            color=color,
            label=f"Rate: {scale:.1f}",
        )

    ax["values"].set_title("True data", fontsize="medium")
    ax["trends"].set_title("Linear fit", fontsize="medium")
    ax["values"].set_ylabel("Relative RMSE (normalised)", fontsize="medium")

    for a in [ax["values"], ax["trends"]]:
        a.grid(which="minor", linestyle="dotted")
        a.set_xlabel("Step count (normalised)")
        a.legend(fontsize="xx-small", ncols=2)
    plt.show()


def solver_scipy(*, method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return np.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    time_span = np.asarray([0.0, 50.0])

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
        return jnp.asarray(solution.y[..., -1])

    return param_to_solution


def solver_probdiffeq(num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf_probdiffeq(y, /, *, t):  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return jnp.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    t0, t1 = (0.0, 50.0)

    @jax.jit
    def param_to_solution(tol):
        # Do inside the function so we jit the Taylor code
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf_probdiffeq, (u0,), t=t0)

        # Build a solver
        ssm = probdiffeq.state_space_model_dense()
        iwp = ssm.prior_wiener_integrated(tcoeffs)
        strategy = probdiffeq.strategy_filter()
        ts = ssm.constraint_ode_ts1(vf_probdiffeq)
        solver = probdiffeq.solver(strategy=strategy, constraint=ts)
        error = probdiffeq.error_residual_std(constraint=ts)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control
        )

        # Solve
        dt0 = ivpsolve.dt0(vf_probdiffeq, (u0,), t=t0)
        solution = solve(iwp, t0=t0, t1=t1, dt0=dt0, atol=1e-2 * tol, rtol=tol)

        # Return the terminal value
        return {
            "precision": precision_fun(solution.u.mean[0]),
            "num_steps": solution.num_steps,
        }

    return param_to_solution


def workprec(fun, *, precision_fun: Callable, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function into parameter-to-workprecision."""

    def parameter_list_to_workprecision(list_of_args, /):
        works_num_steps = []
        works_min = []
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            _x, num_steps = fun(arg)

            precision = precision_fun(fun(arg)[0].block_until_ready())
            times = timeit_fun(lambda: fun(arg)[0].block_until_ready())  # noqa: B023

            precisions.append(precision)
            works_num_steps.append(num_steps)
            works_min.append(min(times))
            works_mean.append(statistics.mean(times))
            if len(times) > 1:
                works_std.append(statistics.stdev(times))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_min": jnp.asarray(works_min),
            "work_num_steps": jnp.asarray(works_num_steps),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


def smooth(x, y, window=2):
    """Smooth a set of data points to improve visualisation."""
    kernel = jnp.ones((window,)) / window
    x = jnp.convolve(x, kernel, mode="valid")
    y = jnp.convolve(y, kernel, mode="valid")
    return x, y


def linear_trend(x, y):
    """Fit a linear curve through the logarithms of x and y."""
    x = jnp.log10(x)
    y = jnp.log10(y)
    scale, bias = jnp.polyfit(x, y, 1)
    return (10 ** (x), 10 ** (scale * x + bias)), (scale, bias)


main()
