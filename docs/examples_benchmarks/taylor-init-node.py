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

# # Initialisation: Neural ODE

# +
"""Benchmark the initialisation methods on a neural ODE problem."""

import functools
import statistics
import time
import timeit
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import taylor


def main(max_time=0.55, repeats=2):
    """Run the script."""
    # Set JAX config
    jax.config.update("jax_enable_x64", True)

    algorithms = {
        r"Forward-mode": odejet_via_jvp(),
        r"Taylor-mode (scan)": taylor_mode_scan(),
        r"Taylor-mode (unroll)": taylor_mode_unroll(),
        r"Taylor-mode (doubling)": taylor_mode_doubling(),
    }

    # Compute a reference solution
    timeit_fun = timeit_fun_from_args(repeats=repeats)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in algorithms.items():
        print("\n")
        print(label)
        results[label] = adaptive_benchmark(
            algo, timeit_fun=timeit_fun, max_time=max_time
        )

    _fig, (axis_perform, axis_compile) = plt.subplots(
        ncols=2, figsize=(8, 3), dpi=150, sharex=True, sharey=True
    )

    for label, wp in results.items():
        inputs = wp["arguments"]
        work_compile = wp["work_compile"]
        work_mean, work_std = wp["work_mean"], wp["work_std"]

        if "doubling" in label:
            num_repeats = jnp.diff(jnp.concatenate((jnp.ones((1,)), inputs)))
            inputs = jnp.arange(1, jnp.amax(inputs) * 1)
            work_compile = _adaptive_repeat(work_compile, num_repeats)
            work_mean = _adaptive_repeat(work_mean, num_repeats)
            work_std = _adaptive_repeat(work_std, num_repeats)

        axis_compile.semilogy(inputs, work_compile, label=label)
        axis_perform.semilogy(inputs, work_mean, label=label)

    axis_compile.set_title("Compilation time")
    axis_perform.set_title("Evaluation time")
    axis_perform.legend(fontsize="small")
    axis_compile.legend(fontsize="small")
    axis_compile.set_xlabel("Number of Derivatives")
    axis_perform.set_xlabel("Number of Derivatives")
    axis_perform.set_ylabel("Wall time (sec)")
    axis_perform.grid(linestyle="dotted")
    axis_compile.grid(linestyle="dotted")

    plt.tight_layout()
    plt.show()


def _adaptive_repeat(xs, ys):
    """Repeat the doubling values correctly to create a comprehensible plot."""
    zs = []
    for x, y in zip(xs, ys):
        zs.extend([x] * int(y))
    return jnp.asarray(zs)


def timeit_fun_from_args(*, repeats: int) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=repeats))

    return timer


def taylor_mode_scan() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = taylor.odejet_padded_scan(vf_auto, (u0,), num=num)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_unroll() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = taylor.odejet_unroll(vf_auto, (u0,), num=num)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_doubling() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = taylor.odejet_doubling_unroll(vf_auto, (u0,), num_doublings=num)
        return jnp.asarray(tcoeffs)

    return estimate


def odejet_via_jvp() -> Callable:
    """Forward-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        tcoeffs = taylor.odejet_via_jvp(vf_auto, (u0,), num=num)
        return jnp.asarray(tcoeffs)

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
    """Benchmark a function iteratively until a max-time threshold is exceeded."""
    work_compile = []
    work_mean = []
    work_std = []
    arguments = []

    t0 = time.perf_counter()
    arg = 1
    while (elapsed := time.perf_counter() - t0) < max_time:
        print(f"num = {arg} | elapsed = {elapsed:.2f} | max_time = {max_time}")
        t0 = time.perf_counter()
        tcoeffs = fun(arg).block_until_ready()
        t1 = time.perf_counter()
        time_compile = t1 - t0

        time_execute = timeit_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

        arguments.append(len(tcoeffs))
        work_compile.append(time_compile)
        work_mean.append(statistics.mean(time_execute))
        work_std.append(statistics.stdev(time_execute))
        arg += 1
    print(f"num = {arg} | elapsed = {elapsed:.2f} | max_time = {max_time}")
    return {
        "work_mean": jnp.asarray(work_mean),
        "work_std": jnp.asarray(work_std),
        "work_compile": jnp.asarray(work_compile),
        "arguments": jnp.asarray(arguments),
    }


main()
