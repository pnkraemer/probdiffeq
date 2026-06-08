"""Taylor recursions | Neural ODE."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import probdiffeq
from probdiffeq.util.benchmark_util import (
    _adaptive_repeat,
    adaptive_benchmark,
    setup_timeit,
)

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(max_time=0.55, repeats=2) -> None:
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
    timeit_fun = setup_timeit(repeats=repeats)

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


def taylor_mode_scan() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_unroll() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_doubling() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_doubling_unroll(num_doublings=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def odejet_via_jvp() -> Callable:
    """Forward-mode estimation."""
    vf_auto, (u0,) = _node()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_via_jvp(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
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

    @probdiffeq.ode
    def vf(x, *, t):
        del t
        for (w1, w2), b1, b2 in zip(weights, biases1, biases2):
            x = fun(w2.T @ fun(w1 @ x + b1) + b2)
        return x

    return vf, (u0,)


main()
