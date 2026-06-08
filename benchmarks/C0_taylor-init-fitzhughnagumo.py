"""Taylor recursions | FitzHugh-Nagumo."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(max_time=0.25, repeats=2) -> None:
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
    timeit_fun = benchmark_util.setup_timeit(repeats=repeats)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in algorithms.items():
        print("\n")
        print(label)
        results[label] = benchmark_util.adaptive_benchmark(
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
            work_compile = benchmark_util.adaptive_repeat(work_compile, num_repeats)
            work_mean = benchmark_util.adaptive_repeat(work_mean, num_repeats)
            work_std = benchmark_util.adaptive_repeat(work_std, num_repeats)

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
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_unroll() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def taylor_mode_doubling() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_doubling_unroll(num_doublings=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def odejet_via_jvp() -> Callable:
    """Forward-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_via_jvp(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)

    return estimate


def _fitzhugh_nagumo():
    u0 = jnp.asarray([-1.0, 1.0])

    @probdiffeq.ode
    def vf_probdiffeq(u, *, t):
        """FitzHugh--Nagumo model."""
        del t
        a, b, c = 0.2, 0.2, 3.0
        du1 = c * (u[0] - u[0] ** 3 / 3 + u[1])
        du2 = -(1.0 / c) * (u[0] - a - b * u[1])
        return jnp.asarray([du1, du2])

    return vf_probdiffeq, (u0,)


main()
