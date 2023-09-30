"""HIRES benchmark.

Run with "python run_benchmark.py 1 10" for thorough marks
and with "python run_benchmark.py 1 1" for a dry-run.

"""
import argparse
import functools
import os
import statistics
import timeit

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
import tqdm
from diffeqzoo import backend, ivps
from jax import config

from probdiffeq import adaptive, ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from probdiffeq.util.doc_util import info


def set_config():
    """Set up the configuration."""
    # x64 precision
    config.update("jax_enable_x64", True)

    # CPU
    config.update("jax_platform_name", "cpu")

    # IVP examples in JAX
    if not backend.has_been_selected:
        backend.select("jax")


def print_info():
    """Print the environment info for this benchmark."""
    # Which version of each software are we using?
    info.print_info()
    print("\n------------------------------------------\n")


def argparse_tolerances():
    """Parse the tolerance from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arange_start", type=int)
    parser.add_argument("arange_stop", type=int)
    args = parser.parse_args()
    return 0.1 ** jnp.arange(args.arange_start, args.arange_stop, step=1.0)


def make_ivp():
    """Create the IVP test-problem."""
    f, u0, (t0, t1), f_args = ivps.hires()

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return f(x, *f_args)

    return vf, u0, t0, t1


def probdiffeq_setup(vf, u0, t0, t1, /, *, num_derivatives):
    """Set the solve() function for ProbDiffEq up."""

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


def scipy_setup(vf, u0, t0, t1, /, *, method):
    """Set the solve() function for SciPy up."""

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


def rmse_relative(expected, *, nugget=1e-5):
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        error_relative = error_absolute / jnp.abs(nugget + received)
        return jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

    return rmse


def param_to_workprecision(fun, precision):
    """Turn a parameter-to-solution function to a parameter-to-workprecision function.

    Turn a function param->solution into a function

    (param1, param2, ...)->(workprecision1, workprecision2, ...)

    where workprecisionX is a dictionary with keys "work" and "precision".
    """

    def param_to_wp(list_of_args):
        works_mean = []
        works_std = []
        precisions = []
        for args in list_of_args:
            precisions.append(precision(fun(args)))

            times = timeit.repeat(lambda: fun(args), number=1, repeat=10)  # noqa: B023

            times = list(times)
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return param_to_wp


if __name__ == "__main__":
    set_config()

    # Assemble algorithms
    ivp = make_ivp()
    impl.select("dense", ode_shape=(8,))
    algorithms = {
        "TS1(n=3)": probdiffeq_setup(*ivp, num_derivatives=3),
        "TS1(n=5)": probdiffeq_setup(*ivp, num_derivatives=5),
        "SciPy (LSODA)": scipy_setup(*ivp, method="LSODA"),
        "SciPy (Radau)": scipy_setup(*ivp, method="Radau"),
    }

    # Compute a reference solution
    reference = algorithms["SciPy (Radau)"](1e-12)
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    tolerances = argparse_tolerances()
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = param_to_workprecision(algo, precision=precision_fun)
        results[label] = param_to_wp(tolerances)

    jnp.save(os.path.dirname(__file__) + "/results.npy", results)
