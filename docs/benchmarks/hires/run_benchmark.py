"""HIRES benchmark.

Run with "python run_benchmark.py 1 10" for thorough marks
and with "python run_benchmark.py 1 1" for a dry-run.

"""
import argparse
import functools
import os
import statistics
import timeit
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
import tqdm
from diffeqzoo import backend, ivps
from jax import config

from probdiffeq import adaptive, ivpsolve, timestep
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from probdiffeq.util.doc_util import info


def set_jax_config() -> None:
    """Set JAX and other external libraries up."""
    # x64 precision
    config.update("jax_enable_x64", True)

    # CPU
    config.update("jax_platform_name", "cpu")

    # IVP examples in JAX
    if not backend.has_been_selected:
        backend.select("jax")


def set_probdiffeq_config() -> None:
    """Set probdiffeq up."""
    impl.select("dense", ode_shape=(8,))


def print_library_info() -> None:
    """Print the environment info for this benchmark."""
    info.print_info()
    print("\n------------------------------------------\n")


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--repeats", type=int)
    return parser.parse_args()


def tolerances_from_args(arguments: argparse.Namespace, /) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(arguments.start, arguments.stop, step=1.0)


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def ivp_hires() -> tuple[Callable, jax.Array, float, float]:
    """Create the IVP test-problem."""
    f, u0, (t0, t1), f_args = ivps.hires()

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return f(x, *f_args)

    return vf, u0, t0, t1


def solver_probdiffeq(vf, u0, t0, t1, /, *, num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        ibm = priors.ibm_adaptive(num_derivatives=num_derivatives)
        ts1 = corrections.ts1()
        strategy = filters.filter_adaptive(ibm, ts1)
        solver = calibrated.dynamic(strategy)
        adaptive_solver = adaptive.adaptive(solver, atol=1e-3 * tol, rtol=tol)

        # Initial state
        vf_auto = functools.partial(vf, t=t0)
        tcoeffs = autodiff.taylor_mode(vf_auto, (u0,), num=num_derivatives)
        init = solver.initial_condition(tcoeffs, output_scale=1.0)

        # Solve
        dt0 = timestep.initial(vf_auto, (u0,))
        solution = ivpsolve.simulate_terminal_values(
            vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
        )

        # Return the terminal value
        return jax.block_until_ready(solution.u)

    return param_to_solution


def solver_scipy(vf, u0, t0, t1, /, *, method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""
    # todo: should the whole problem be transformed into a numpy array?

    def vf_scipy(t, u):
        return vf(u, t=t)

    u0 = np.asarray(u0)
    time_span = np.asarray([np.asarray(t0), np.asarray(t1)])

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
        return solution.y[:, -1]

    return param_to_solution


def rmse_relative(expected: jax.Array, *, nugget=1e-5) -> Callable:
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        error_relative = error_absolute / jnp.abs(nugget + received)
        return jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

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

    # Read configuration from command line
    args = parse_arguments()
    tolerances = tolerances_from_args(args)
    timeit_fun = timeit_fun_from_args(args)

    # Assemble algorithms
    hires = ivp_hires()
    algorithms = {
        r"TS1($\nu=3$)": solver_probdiffeq(*hires, num_derivatives=3),
        r"TS1($\nu=5$)": solver_probdiffeq(*hires, num_derivatives=5),
        "SciPy (LSODA)": solver_scipy(*hires, method="LSODA"),
        "SciPy (Radau)": solver_scipy(*hires, method="Radau"),
    }

    # Compute a reference solution (assert that warning is raise because
    with testing.warns():
        reference = algorithms["SciPy (LSODA)"](1e-15)
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    # Save results
    jnp.save(os.path.dirname(__file__) + "/results.npy", results)
