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

# # WP diagram: Robertson (ODE vs DAE)

# +
"""Work-precision diagram on the Robertson problem.

The Robertson problem is interesting for many reasons:
- It comes in DAE, MM-ODE, and ODE form
  so we can compare different information operators
- It has an exponential timescale so (good) adaptive
  steps are needed; fixed steps are hopeless.
- Its y-states have wildly different scales,
  so a good prior model is important.
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


def main(start=2.0, stop=12.0, step=1.0, repeats=2, time_span=(1e-6, 1e5)):
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    t0, t1 = time_span
    save_at = jnp.exp(jnp.linspace(jnp.log(t0), jnp.log(t1), num=100))
    ts, ys = solve_ivp_once(save_at=save_at, tol=1e-10, method="LSODA")

    _fig, ax = plt.subplots(nrows=3, figsize=(5, 5), sharex=True)
    ax[0].set_title("Robertson solution")
    ax[0].semilogx(ts, 1e-10 + ys[:, 0])
    ax[1].semilogx(ts, 1e-10 + ys[:, 1])
    ax[2].semilogx(ts, 1e-10 + ys[:, 2])

    ax[0].set_ylabel("State $y_1$")
    ax[1].set_ylabel("State $y_2$")
    ax[2].set_ylabel("State $y_3$")
    ax[2].set_xlabel("Time $t$")
    ax[0].set_xlim((t0, t1))
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {
        "DAE | Jet": solver_dae(num_derivatives=4, time_span=time_span),
        "ODE | TS1": solver_ode(num_derivatives=4, time_span=time_span),
        "ODE | BDF (Scipy)": solver_scipy(method="BDF", time_span=time_span),
        "ODE | LSODA (Scipy)": solver_scipy(method="LSODA", time_span=time_span),
        "ODE | Radau (Scipy)": solver_scipy(method="Radau", time_span=time_span),
    }

    # Compute a reference solution
    reference = solver_scipy(method="Radau", time_span=time_span)(0.1 * tolerances[-1])
    rmse_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = workprec(algo, precision_fun=rmse_fun, work_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)
    _fig, ax = plt.subplots(ncols=2, figsize=(13, 5))

    for label, wp in results.items():
        wdw = 2  # window

        precision, y = wp["precision"], wp["work_mean"]
        x, _ = precision.T
        x = jnp.exp(jnp.convolve(jnp.log(x), jnp.ones((wdw,)) / wdw, mode="valid"))
        y = jnp.exp(jnp.convolve(jnp.log(y), jnp.ones((wdw,)) / wdw, mode="valid"))
        ax[0].loglog(x, y, label=label)

        x, size = precision.T
        eps = jnp.finfo(x.dtype).eps
        ax[1].loglog(tolerances, eps + jnp.abs(size - 1.0), "-", label=label)

    ax[0].set_title("Work-precision diagram")
    ax[0].set_xlabel("Precision (relative RMSE)")
    ax[0].set_ylabel("Work (avg. wall time)")
    ax[0].grid(linestyle="dotted", which="both")
    ax[0].legend(fontsize="small")

    ax[1].set_title("Constraint violation")
    ax[1].set_xlabel("Tolerance (user input)")
    ax[1].set_ylabel("Algebraic constraint violation")
    ax[1].grid(linestyle="dotted", which="both")
    ax[1].legend(fontsize="small")

    plt.tight_layout()
    plt.show()


def solve_ivp_once(*, save_at, method, tol):
    """Compute plotting-values for the IVP."""

    def vf(t, y):
        del t
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = jnp.array([1.0, 0.0, 0.0])

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
    return solution.t, solution.y.T


def setup_tolerances(*, start: float, stop: float, step: float) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(start, stop, step=step)


def setup_timeit(*, repeats: int) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=repeats))

    return timer


def solver_ode(*, num_derivatives: int, time_span) -> Callable:
    """Construct a method that solves Robertson as an ODE."""

    def root(u, du, /, *, t):
        return du - vf(u, t=t)

    def vf(y, *, t):
        del t
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

        base_scale = jnp.diag(jnp.asarray([1e0, 1e-4, 1e-1]))
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(
            tcoeffs, output_scale=base_scale
        )
        ts = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
        strategy = probdiffeq.strategy_filter(ssm=ssm)

        solver = probdiffeq.solver(strategy=strategy, prior=ibm, constraint=ts, ssm=ssm)
        error = probdiffeq.error_state_std(constraint=ts, prior=ibm, ssm=ssm)

        control = ivpsolve.control_integral()

        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_dae(*, num_derivatives: int, time_span) -> Callable:
    """Construct a method that solves Robertson as a DAE."""

    def root(u, du, /, *, t):
        del t
        return [vf_differential(u, du), vf_algebraic(u)]

    def vf_differential(y, du):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        return jnp.stack([du[0] - f0, du[1] - f1])

    def vf_algebraic(u):
        return u[0] + u[1] + u[2] - 1

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        t0, t1 = time_span
        y0 = jnp.array([1.0, 0.0, 0.0])

        # This initial uncertainty encodes that the third variable
        # in the initial condition is actually redundant. If we were to
        # put zero, the solver would fail because it tries to conditoin
        # something that is already certain
        # TODO: create a prior_wiener_integrated_from_initcond or so
        #       that fills all thos taylor coefficients with zero and so on.
        #       This could also be the place where we set
        #       "differential_variable" like in the TODO below
        #       so that DAEs can be solved conveniently
        eps = 10 * jnp.finfo(y0.dtype).eps
        init_unc = jnp.asarray([0.0, 0.0, eps])
        zeros, ones = jnp.zeros_like(y0), jnp.ones_like(y0)
        tcoeffs = [y0, *[zeros for _ in range(num_derivatives)]]
        tcoeffs_std = [init_unc, *[ones for _ in range(num_derivatives)]]

        # This base scale is also critical to Robertson, because
        # the solutions live on vastly different scales
        # (but don't vary much within these scales). Priming the output
        # scale like this is really beneficial for the solver.
        # TODO: what is the best "prime" for the solver?
        #       this should be an expectation-maximisation thing right?
        base_scale = jnp.diag(jnp.asarray([1e0, 1e-4, 1e-1]))
        init, ibm, ssm = probdiffeq.prior_wiener_integrated(
            tcoeffs, tcoeffs_std=tcoeffs_std, output_scale=base_scale
        )

        # TODO: Give all root-constraints some argument like
        #       differential_variable=[True, True, False] or so
        #       like in DifferentialEquations.jl, where for the
        #       variables that are not differential, the marginal
        #       STD in the initialisation step is inflated
        #       by a machine epsilon to avoid dividing by zero.
        #       Currently, a user needs to do this manually.
        #       (Or document this really well?)
        ts = probdiffeq.constraint_root_jet(root, ssm=ssm)
        strategy = probdiffeq.strategy_filter(ssm=ssm)

        # For proper DAEs, non-iterated solver's simply don't cut it
        eps = 10 * jnp.finfo(y0.dtype).eps
        solver = probdiffeq.solver_iterated(
            strategy=strategy,
            prior=ibm,
            constraint=ts,
            ssm=ssm,
            constraint_init=ts,  # Critical for DAEs
            tol=eps,
        )

        # The state-error-estimate doesn't care about the dimension
        # of the DAE, which is exactly what we need here
        error = probdiffeq.error_state_std(constraint=ts, prior=ibm, ssm=ssm)

        # Integral controllers just work better than proportional-integral ones
        # TODO: build PID controllers (is this "gustafsson"?) for iterated solvers?
        control = ivpsolve.control_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )
        # TODO: how do I manage to solve this thing without damping?
        # Where are we NaN'ing out? Is it because du2 is not observed?
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, time_span) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf(t, y):
        del t
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

        error_relative = error_absolute / (1e-5 + jnp.abs(expected))
        rmse = jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

        algebraic = jnp.sum(received)
        return rmse, algebraic

    return rmse


def workprec(fun, *, precision_fun: Callable, work_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function into parameter-to-workprecision."""

    def parameter_list_to_workprecision(list_of_args, /):
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg).block_until_ready())
            work = work_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

            precisions.append(precision)
            works_mean.append(statistics.mean(work))
            works_std.append(statistics.stdev(work))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


main()
