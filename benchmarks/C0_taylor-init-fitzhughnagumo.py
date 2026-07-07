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


# Set JAX config
jax.config.update("jax_enable_x64", True)


def main(max_time=0.75, repeats=1) -> None:
    """Run the script."""
    algorithms = {
        r"Forward-mode": odejet_via_jvp(),
        r"Taylor-mode (scan)": taylor_mode_scan(),
        r"Taylor-mode (unroll)": taylor_mode_unroll(),
        r"Taylor-mode (doubling)": taylor_mode_doubling(),
    }

    # Compute all work-precision diagrams
    _fig, (axis_compile, axis_perform) = plt.subplots(
        ncols=2, figsize=(8, 3), constrained_layout=True, dpi=120, sharex=True
    )

    list_of_args = list(range(1, 20))
    for label, algo in algorithms.items():
        print("\n")
        print(label)
        workprec = benchmark_util.workprec(
            algo, time_max_sec=max_time, print_progress=True, num_timing_calls=repeats
        )
        wp = workprec(list_of_args)
        inputs = wp.arg
        work_compile = wp.work_first_run
        work_mean = wp.work.mean(axis=-1)

        if "doubling" in label:
            num_repeats = 2**inputs
            work_compile = benchmark_util.adaptive_repeat(work_compile, num_repeats)
            work_mean = benchmark_util.adaptive_repeat(work_mean, num_repeats)
            inputs = jnp.arange(1, 2 ** (jnp.amax(inputs) + 1) - 1)

        axis_compile.semilogy(inputs, work_compile, label=label)
        axis_perform.semilogy(inputs, work_mean, label=label)

    axis_compile.set_title("Compilation time")
    axis_perform.set_title("Evaluation time")
    axis_compile.set_ylabel("Wall time (sec)")
    for ax in [axis_perform, axis_compile]:
        ax.legend(fontsize="x-small")
        ax.set_xlabel("Number of Derivatives")
        ax.grid(linestyle="dotted")

    plt.show()


def taylor_mode_scan() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)[-1]

    return estimate


def taylor_mode_unroll() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)[-1]

    return estimate


def taylor_mode_doubling() -> Callable:
    """Taylor-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_doubling_unroll(num_doublings=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)[-1]

    return estimate


def odejet_via_jvp() -> Callable:
    """Forward-mode estimation."""
    vf_auto, (u0,) = _fitzhugh_nagumo()

    @functools.partial(jax.jit, static_argnames=["num"])
    def estimate(num):
        jetexpand = probdiffeq.jetexpand_ode_via_jvp(num=num)
        tcoeffs, _ = jetexpand(vf_auto, (u0,), t=0.0)
        return jnp.asarray(tcoeffs)[-1]

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
