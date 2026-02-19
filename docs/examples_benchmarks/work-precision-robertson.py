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

# # WP diagram: Stiff van-der-Pol

# +
"""Work-precision diagram on the Robertson problem.

The robertson problem is interesting for many reasons:
- It comes in DAE, MM-ODE, and ODE form so we can compare different information operators
- It has an exponential timescale so (good) adaptive steps are needed, fixed steps are hopeless
- Its y-states have wildly different scales, so a good prior model is important
"""

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


def main(start=3.0, stop=12.0, step=0.5, repeats=2, time_span=(1e-6, 1e3)):
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    t0, t1 = time_span
    save_at = jnp.exp(jnp.linspace(jnp.log(t0), jnp.log(t1), num=100))
    ts, ys = solve_ivp_once(save_at=save_at, method="LSODA")

    solver = solver_probdiffeq_plot(num_derivatives=4, save_at=save_at)
    t, mean, std = jax.jit(solver)(1e-12)

    _fig, ax = plt.subplots(nrows=3, figsize=(8, 8))
    ax[0].semilogx(ts, ys[:, 0])
    ax[1].semilogx(ts, ys[:, 1])
    ax[2].semilogx(ts, ys[:, 2])

    ax[0].semilogx(t, mean[:, 0])
    ax[1].semilogx(t, mean[:, 1])
    ax[2].semilogx(t, mean[:, 2])

    ax[0].set_title(rmse_relative(expected=ys)(received=mean))
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {
        "Old: TS1(4)": solver_probdiffeq(num_derivatives=4, time_span=time_span),
        "Old: TS1(7)": solver_probdiffeq(num_derivatives=7, time_span=time_span),
    }

    # Compute a reference solution
    reference = solver_scipy(method="Radau", time_span=time_span)(0.1 * tolerances[-1])
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)
    _fig, ax = plt.subplots(figsize=(8, 5))

    for label, wp in results.items():
        wdw = 1  # window
        x, y = wp["precision"], wp["work_mean"]
        x = jnp.exp(jnp.convolve(jnp.log(x), jnp.ones((wdw,)) / wdw, mode="valid"))
        y = jnp.exp(jnp.convolve(jnp.log(y), jnp.ones((wdw,)) / wdw, mode="valid"))
        ax.loglog(x, y, label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small")

    plt.tight_layout()
    plt.show()


def solve_ivp_once(*, save_at, method):
    """Compute plotting-values for the IVP."""

    def vf(t, y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = jnp.array([1.0, 0.0, 0.0])

    tol = 1e-12
    t0, t1 = save_at[0], save_at[-1]
    solution = scipy.integrate.solve_ivp(
        vf,
        y0=y0,
        t_span=(t0, t1),
        t_eval=save_at,
        atol=1e-3 * tol,
        rtol=tol,
        method=method,
    )
    print("Baseline:", solution.nfev)
    return solution.t, solution.y.T


def setup_tolerances(*, start: float, stop: float, step: float) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(start, stop, step=step)


def setup_timeit(*, repeats: int) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=repeats))

    return timer


def solver_probdiffeq_plot(*, num_derivatives: int, save_at) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    def root(u, du, /, *, t):
        return du - vf(u, t=t)

    def vf(y, *, t):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

    t0, t1 = save_at[0], save_at[-1]
    y0 = jnp.array([1.0, 0.0, 0.0])

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        vf_auto = functools.partial(vf, t=t0)
        tcoeffs = taylor.odejet_padded_scan(vf_auto, (y0,), num=num_derivatives - 1)

        # Very important for Robertson: anisotropic output scales
        base_scale = jnp.diag(jnp.asarray([1.0, jnp.sqrt(1e-5), 1.0]))
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(
            tcoeffs, output_scale=base_scale
        )
        ts = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
        strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)

        # No need for dynamic solvers because the output scales don't vary much
        solver = probdiffeq.solver_iterated(
            strategy=strategy, prior=ibm, constraint=ts, ssm=ssm
        )
        error = probdiffeq.error_state_std(constraint=ts, prior=ibm, ssm=ssm)

        control = ivpsolve.control_integral()

        solve = ivpsolve.solve_adaptive_save_at(
            solver=solver, error=error, control=control, clip_dt=True
        )
        solution = solve(init, save_at=save_at, atol=1e-3 * tol, rtol=tol)
        jax.debug.print("{}", solution.num_steps)
        return solution.t, solution.u.mean[0], solution.u.std[0]

    return param_to_solution


def solver_probdiffeq(*, num_derivatives: int, time_span) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    def root(u, du, /, *, t):
        return du - vf(u, t=t)

    def vf(y, *, t):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

    t0, t1 = time_span
    y0 = jnp.array([1.0, 0.0, 0.0])

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        vf_auto = functools.partial(vf, t=t0)
        tcoeffs = taylor.odejet_padded_scan(vf_auto, (y0,), num=num_derivatives - 1)

        base_scale = jnp.diag(jnp.asarray([1.0, jnp.sqrt(1e-5), 1.0]))
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(
            tcoeffs, output_scale=base_scale
        )
        ts = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
        strategy = probdiffeq.strategy_filter(ssm=ssm)

        solver = probdiffeq.solver_iterated(
            strategy=strategy, prior=ibm, constraint=ts, ssm=ssm
        )
        error = probdiffeq.error_state_std(constraint=ts, prior=ibm, ssm=ssm)

        control = ivpsolve.control_integral()

        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)
        jax.debug.print("{}", solution.num_steps)
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, time_span) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf(t, y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = jnp.array([1.0, 0.0, 0.0])

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            vf,
            y0=y0,
            t_span=time_span,
            t_eval=time_span,
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return jnp.asarray(solution.y[:, -1])

    return param_to_solution


def rmse_relative(expected: jax.Array) -> Callable:
    """Compute the absolute RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        # print(error_absolute)
        error_relative = error_absolute / (1e-5 + jnp.abs(expected))
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
