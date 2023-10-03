"""Benchmark the initialisation methods on Pleiades.

See makefile for instructions.
"""
import argparse
import functools
import os
import statistics
import time
import timeit
from typing import Callable

import jax
import jax.numpy as jnp
from jax import config

from probdiffeq.impl import impl
from probdiffeq.solvers.taylor import autodiff
from probdiffeq.util.doc_util import info


def set_jax_config() -> None:
    """Set JAX and other external libraries up."""
    # x64 precision
    config.update("jax_enable_x64", True)

    # CPU
    config.update("jax_platform_name", "cpu")


def set_probdiffeq_config() -> None:
    """Set probdiffeq up."""
    impl.select("isotropic", ode_shape=(14,))


def print_library_info() -> None:
    """Print the environment info for this benchmark."""
    info.print_info()
    print("\n------------------------------------------\n")


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_time_per_method", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def taylor_mode() -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""
    # fmt: off
    u0 = jnp.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
        ]
    )
    du0 = jnp.asarray(
        [
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    @jax.jit
    def vf_probdiffeq(u, du, *, t):  # noqa: ARG001
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = jnp.arange(1, 8)[None, :]
        ddx = jnp.sum(jnp.nan_to_num(mj * (xj - xi) / rij), axis=1)
        ddy = jnp.sum(jnp.nan_to_num(mj * (yj - yi) / rij), axis=1)
        return jnp.concatenate((ddx, ddy))

    t0 = 0.0

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        vf_auto = functools.partial(vf_probdiffeq, t=t0)
        tcoeffs = autodiff.taylor_mode(vf_auto, (u0, du0), num=num - 1)
        return jax.block_until_ready(tcoeffs)

    return estimate


def workprec(fun, *, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function to a parameter-to-workprecision function.

    Turn a function param->solution into a function

    (param1, param2, ...)->(workprecision1, workprecision2, ...)

    where workprecisionX is a dictionary with keys "work" and "precision".
    """

    def parameter_list_to_workprecision(max_time=1.0, /):
        work_compile = []
        work_mean = []
        work_std = []
        arguments = []

        t0 = time.perf_counter()
        arg = 1
        while (elapsed := time.perf_counter() - t0) < max_time:
            print(arg, elapsed)
            t0 = time.perf_counter()
            _ = fun(arg)
            t1 = time.perf_counter()
            time_compile = t1 - t0

            time_execute = timeit_fun(lambda: fun(arg))  # noqa: B023

            arguments.append(arg)
            work_compile.append(time_compile)
            work_mean.append(statistics.mean(time_execute))
            work_std.append(statistics.stdev(time_execute))
            arg += 1
        return {
            "work_mean": jnp.asarray(work_mean),
            "work_std": jnp.asarray(work_std),
            "work_compile": jnp.asarray(work_compile),
            "arguments": jnp.asarray(arguments),
        }

    return parameter_list_to_workprecision


if __name__ == "__main__":
    set_jax_config()
    algorithms = {
        r"Taylor-mode": taylor_mode(),
    }

    # Compute a reference solution
    args = parse_arguments()
    timeit_fun = timeit_fun_from_args(args)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in algorithms.items():
        param_to_wp = workprec(algo, timeit_fun=timeit_fun)
        results[label] = param_to_wp(args.max_time_per_method)

    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")

    print("add things to the makefile, and get other initialisers into it.")
