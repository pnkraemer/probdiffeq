"""Van-der-Pol benchmark.

See makefile for instructions.
"""
import argparse
import functools
import os
import statistics
import timeit
from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
import tqdm
from jax import config

from probdiffeq import adaptive, controls, ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from probdiffeq.util.doc_util import info


def set_jax_config() -> None:
    """Set JAX and other external libraries up."""
    # x64 precision
    config.update("jax_enable_x64", True)

    # CPU
    config.update("jax_platform_name", "cpu")


def set_probdiffeq_config() -> None:
    """Set probdiffeq up."""
    impl.select("dense", ode_shape=(1,))


def print_library_info() -> None:
    """Print the environment info for this benchmark."""
    info.print_info()
    print("\n------------------------------------------\n")


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def tolerances_from_args(arguments: argparse.Namespace, /) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(arguments.start, arguments.stop, step=1.0)


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def solver_probdiffeq(*, num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @jax.jit
    def vf_probdiffeq(u, du, *, t):  # noqa: ARG001
        """Van-der-Pol dynamics as a second-order differential equation."""
        return 1e5 * ((1.0 - u**2) * du - u)

    t0, t1 = 0.0, 3.0
    u0, du0 = (jnp.atleast_1d(2.0), jnp.atleast_1d(0.0))
    t0, t1 = (0.0, 6.3)

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        ibm = priors.ibm_adaptive(num_derivatives=num_derivatives)
        ts0_or_ts1 = corrections.ts1(ode_order=2)
        strategy = filters.filter_adaptive(ibm, ts0_or_ts1)
        solver = calibrated.dynamic(strategy)
        control = controls.proportional_integral_clipped()
        adaptive_solver = adaptive.adaptive(
            solver, atol=1e-3 * tol, rtol=tol, control=control
        )

        # Initial state
        vf_auto = functools.partial(vf_probdiffeq, t=t0)
        tcoeffs = autodiff.taylor_mode_scan(vf_auto, (u0, du0), num=num_derivatives - 1)
        init = solver.initial_condition(tcoeffs, output_scale=1.0)

        # Solve
        dt0 = timestep.initial(vf_auto, (u0, du0))
        solution = ivpsolve.simulate_terminal_values(
            vf_probdiffeq, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
        )

        # Return the terminal value
        return jax.block_until_ready(solution.u)

    return param_to_solution


def solver_diffrax(*, solver) -> Callable:
    """Construct a solver that wraps Diffrax' solution routines."""

    @diffrax.ODETerm
    @jax.jit
    def vf_diffrax(_t, u, _args):
        """Van-der-Pol dynamics as a first-order differential equation."""
        return jnp.asarray([u[1], 1e5 * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    t0, t1 = 0.0, 3.0
    u0 = jnp.concatenate((jnp.atleast_1d(2.0), jnp.atleast_1d(0.0)))
    t0, t1 = (0.0, 6.3)

    @jax.jit
    def param_to_solution(tol):
        controller = diffrax.PIDController(atol=1e-3 * tol, rtol=tol)
        saveat = diffrax.SaveAt(t0=False, t1=True, ts=None)
        solution = diffrax.diffeqsolve(
            vf_diffrax,
            y0=u0,
            t0=t0,
            t1=t1,
            saveat=saveat,
            stepsize_controller=controller,
            dt0=None,
            max_steps=10_000,
            solver=solver,
        )
        return jax.block_until_ready(solution.ys[0, 0])

    return param_to_solution


def solver_scipy(method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, u):
        """Van-der-Pol dynamics as a first-order differential equation."""
        return np.asarray([u[1], 1e5 * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    u0 = np.concatenate((np.atleast_1d(2.0), np.atleast_1d(0.0)))
    time_span = np.asarray((0.0, 6.3))

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
        return solution.y[0, -1]

    return param_to_solution


def plot_ivp_solution():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, u):
        """Van-der-Pol dynamics as a first-order differential equation."""
        return np.asarray([u[1], 1e5 * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    u0 = np.concatenate((np.atleast_1d(2.0), np.atleast_1d(0.0)))
    time_span = np.asarray((0.0, 6.3))

    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def rmse_absolute(expected: jax.Array) -> Callable:
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        return jnp.linalg.norm(error_absolute) / jnp.sqrt(error_absolute.size)

    return rmse


def workprec(fun, *, precision_fun: Callable, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function to a parameter-to-workprecision function.

    Turn a function param->solution into a function

    (param1, param2, ...)->(workprecision1, workprecision2, ...)

    where workprecisionX is a dictionary with keys "work" and "precision".
    """

    def parameter_list_to_workprecision(list_of_args, /):
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg))
            times = timeit_fun(lambda: fun(arg))  # noqa: B023

            precisions.append(precision)
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


if __name__ == "__main__":
    # Set up all the configs
    set_jax_config()
    set_probdiffeq_config()
    print_library_info()

    # Simulate once to get plotting code
    ts, ys = plot_ivp_solution()

    # Read configuration from command line
    args = parse_arguments()
    tolerances = tolerances_from_args(args)
    timeit_fun = timeit_fun_from_args(args)

    # Assemble algorithms
    algorithms = {
        "SciPy: 'Radau'": solver_scipy(method="Radau"),
        "SciPy: 'LSODA'": solver_scipy(method="LSODA"),
        "Diffrax: Kvaerno5()": solver_diffrax(solver=diffrax.Kvaerno5()),
        r"ProbDiffEq: TS1($3$)": solver_probdiffeq(num_derivatives=3),
        r"ProbDiffEq: TS1($4$)": solver_probdiffeq(num_derivatives=4),
        r"ProbDiffEq: TS1($5$)": solver_probdiffeq(num_derivatives=5),
    }

    # Compute a reference solution
    reference = solver_probdiffeq(num_derivatives=4)(1e-10)
    precision_fun = rmse_absolute(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        jnp.save(os.path.dirname(__file__) + "/plot_ts.npy", ts)
        jnp.save(os.path.dirname(__file__) + "/plot_ys.npy", ys)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")
