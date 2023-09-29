"""HIRES benchmark."""
import functools
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
from probdiffeq.util.doc_util import info, notebook


def print_info():
    """Print the environment info for this benchmark."""
    # x64 precision
    config.update("jax_enable_x64", True)

    # CPU
    config.update("jax_platform_name", "cpu")

    # IVP examples in JAX
    if not backend.has_been_selected:
        backend.select("jax")

    # Nice-looking plots
    plt.rcParams.update(notebook.plot_config())

    # Which version of each software are we using?
    info.print_info()
    print("\n------------------------------------------\n")


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
        dt0 = timestep.initial_adaptive(
            vf,
            (u0,),
            t0=t0,
            error_contraction_rate=num_derivatives + 1,
            atol=1e-3 * tol,
            rtol=tol,
        )
        solution = ivpsolve.simulate_terminal_values(
            vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
        )

        # Return the terminal value
        return solution.u

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


def rmse_relative(received, /, expected, *, nugget=1e-5):
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)
    received = jnp.asarray(received)
    error_absolute = jnp.abs(expected - received)
    error_relative = error_absolute / jnp.abs(nugget + received)
    return jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)


def param_to_workprecision(fun, precision):
    """Turn a parameter-to-solution function to a parameter-to-workprecision function.

    Turn a function param->solution into a function

    (param1, param2, ...)->(workprecision1, workprecision2, ...)

    where workprecisionX is a dictionary with keys "work" and "precision".
    """

    def param_to_wp(list_of_args):
        works = []
        precisions = []
        for args in list_of_args:
            precisions.append(precision(fun(args)))
            time_t0 = time.time()
            _ = fun(args)
            time_t1 = time.time()
            works.append(time_t1 - time_t0)
        return {"work": jnp.asarray(works), "precision": jnp.asarray(precisions)}

    return param_to_wp


if __name__ == "__main__":
    print_info()
    ivp = make_ivp()

    # Assemble algorithms
    impl.select("dense", ode_shape=(8,))
    algorithms = {
        "ProbDiffEq (low)": probdiffeq_setup(*ivp, num_derivatives=3),
        "ProbDiffEq (high)": probdiffeq_setup(*ivp, num_derivatives=5),
        "SciPy (LSODA)": scipy_setup(*ivp, method="LSODA"),
        "SciPy (Radau)": scipy_setup(*ivp, method="Radau"),
    }
    # Compute a reference solution
    reference = algorithms["SciPy (Radau)"](1e-10)

    def precision_fun(s, /):
        """Evaluate the precision of an estimate."""
        return rmse_relative(s, expected=reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = param_to_workprecision(algo, precision=precision_fun)
        results[label] = param_to_wp(0.1 ** jnp.arange(1.0, 9.0))

    jnp.save(os.path.dirname(__file__) + "/results.npy", results)
