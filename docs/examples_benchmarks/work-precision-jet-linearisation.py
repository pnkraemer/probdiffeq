# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # WP diagram: Jets

# +
"""Jet work-precision diagram."""

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


def main(start=3.0, stop=10.0, step=1.0, repeats=5):
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    ts, ys = solve_ivp_once()

    _fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(ts, ys)
    ax.set_title("Lotka-Volterra problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {}
    idcs = [3, 6, 9]
    for i in idcs:
        algorithms[f"Old: TS1({i})"] = solver_probdiffeq_ts(i)
        algorithms[f"New: Jet({i})"] = solver_probdiffeq_jet(i)

    # Compute a reference solution
    reference = solver_scipy(method="BDF")(1e-13)
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    _fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for label, wp in results.items():
        style = {
            "linestyle": "dashed" if "ts" in label.lower() else "solid",
            "color": "C0" if "3" in label else "C1" if "6" in label else "C2",
        }

        x, y = wp["precision"], wp["work_mean"]
        ax.loglog(x, y, alpha=0.25, **style)

        x_lin, y_lin = linear_trend(x, y)
        ax.loglog(x_lin, y_lin, label=label, **style)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def linear_trend(x, y):
    """Fit a linear curve through the logarithms of x and y."""
    x = jnp.log10(x)
    y = jnp.log10(y)
    scale, bias = jnp.polyfit(x, y, 1)
    return 10 ** (x), 10 ** (scale * x + bias)


def solve_ivp_once():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return np.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    time_span = np.asarray([0.0, 50.0])
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


def solver_probdiffeq_ts(num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    def root(y, dy, *, t):
        return dy - vf(y, t=t)

    def vf(y, /, *, t):  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return jnp.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    t0, t1 = (0.0, 50.0)

    @jax.jit
    def param_to_solution(tol):
        vf_auto = functools.partial(vf, t=t0)
        tcoeffs = taylor.odejet_padded_scan(vf_auto, (u0,), num=num_derivatives)

        init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs)
        strategy = probdiffeq.strategy_filter(ssm=ssm)
        corr = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
        solver = probdiffeq.solver(
            strategy=strategy, prior=ibm, constraint=corr, ssm=ssm
        )
        errorest = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            errorest=errorest, solver=solver, control=control
        )
        dt0 = ivpsolve.dt0(vf_auto, (u0,))

        # Build a solver
        solution = solve(init, t0=t0, t1=t1, dt0=dt0, atol=1e-2 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_probdiffeq_jet(num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    def root(y, dy, *, t):
        return dy - vf(y, t=t)

    def vf(y, /, *, t):  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return jnp.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    t0, t1 = (0.0, 50.0)

    @jax.jit
    def param_to_solution(tol):
        zeros, ones = jnp.zeros_like(u0), jnp.ones_like(u0)
        tcoeffs = [u0, *[zeros for _ in range(num_derivatives)]]
        tcoeffs_std = [zeros, *[ones for _ in range(num_derivatives)]]
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(
            tcoeffs, tcoeffs_std=tcoeffs_std
        )
        strategy = probdiffeq.strategy_filter(ssm=ssm)
        corr = probdiffeq.constraint_root_jet_ts1(root, ssm=ssm)
        solver = probdiffeq.solver(
            strategy=strategy, prior=ibm, constraint=corr, ssm=ssm, update_at_init=True
        )
        error_norm = probdiffeq.errorest_error_norm_rms_then_scale()
        errorest = probdiffeq.errorest_local_residual(
            constraint=corr, prior=ibm, ssm=ssm, error_norm=error_norm
        )

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            errorest=errorest, solver=solver, control=control
        )

        # Build a solver
        solution = solve(init, t0=t0, t1=t1, atol=1e-2 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


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
