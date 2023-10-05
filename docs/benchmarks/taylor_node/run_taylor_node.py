"""Benchmark the initialisation methods on a Neural ODE.

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
from diffeqzoo import backend
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
    parser.add_argument("--max_time", type=float)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def taylor_mode() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = autodiff.taylor_mode(vf_auto, (u0,), num=num)
        return jax.block_until_ready(tcoeffs)

    return estimate


def taylor_mode_unroll() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = autodiff.taylor_mode_unroll(vf_auto, (u0,), num=num)
        return jax.block_until_ready(tcoeffs)

    return estimate


def taylor_mode_doubling() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = autodiff.taylor_mode_doubling(vf_auto, (u0,), num_doublings=num)
        return jax.block_until_ready(tcoeffs)

    return estimate


def forward_mode() -> Callable:
    """Forward-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = autodiff.forward_mode(vf_auto, (u0,), num=num)
        return jax.block_until_ready(tcoeffs)

    return estimate


def _node():
    N = 100
    M = 100
    num_layers = 2

    key = jax.random.PRNGKey(seed=1)
    key1, key2, key3, key4 = jax.random.split(key, num=4)

    u0 = jax.random.uniform(key1, shape=(N,))

    weights = jax.random.normal(key2, shape=(num_layers, 2, M, N))
    biases1 = jax.random.normal(key3, shape=(num_layers, M))
    biases2 = jax.random.normal(key4, shape=(num_layers, N))

    fun = jnp.tanh

    @jax.jit
    def vf(x):
        for (w1, w2), b1, b2 in zip(weights, biases1, biases2):
            x = fun(w2.T @ fun(w1 @ x + b1) + b2)
        return x

    return vf, (u0,)


def adaptive_benchmark(fun, *, timeit_fun: Callable, max_time) -> dict:
    """Call  repeatedly until a time-threshold is exceeded."""
    work_compile = []
    work_mean = []
    work_std = []
    arguments = []

    t0 = time.perf_counter()
    arg = 1
    while (elapsed := time.perf_counter() - t0) < max_time:
        print("num =", arg, "| elapsed =", elapsed, "| max_time =", max_time)
        t0 = time.perf_counter()
        tcoeffs = fun(arg)
        t1 = time.perf_counter()
        time_compile = t1 - t0

        time_execute = timeit_fun(lambda: fun(arg))  # noqa: B023

        arguments.append(len(tcoeffs))
        work_compile.append(time_compile)
        work_mean.append(statistics.mean(time_execute))
        work_std.append(statistics.stdev(time_execute))
        arg += 1
    print("num =", arg, "| elapsed =", elapsed, "| max_time =", max_time)
    return {
        "work_mean": jnp.asarray(work_mean),
        "work_std": jnp.asarray(work_std),
        "work_compile": jnp.asarray(work_compile),
        "arguments": jnp.asarray(arguments),
    }


if __name__ == "__main__":
    set_jax_config()
    backend.select("jax")
    algorithms = {
        r"Forward-mode": forward_mode(),
        r"Taylor-mode (scan)": taylor_mode(),
        r"Taylor-mode (unroll)": taylor_mode_unroll(),
        r"Taylor-mode (doubling)": taylor_mode_doubling(),
    }

    # Compute a reference solution
    args = parse_arguments()
    timeit_fun = timeit_fun_from_args(args)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in algorithms.items():
        print("\n")
        print(label)
        results[label] = adaptive_benchmark(
            algo, timeit_fun=timeit_fun, max_time=args.max_time
        )
    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")
