"""Precompute and save reference solutions. Accelerate testing."""
import diffeqzoo.ivps
import jax.numpy as jnp
from diffeqzoo import backend
from jax.config import config

from probdiffeq.taylor import autodiff


def set_environment():
    """Set the environment (e.g., 64-bit precision).

    The setup used to precompute references should match that of the other tests.
    """
    # Test on CPU.
    config.update("jax_platform_name", "cpu")

    # Double precision
    # Needed for equivalence tests for smoothers.
    config.update("jax_enable_x64", True)

    # IVPs in JAX
    backend.select("jax")


def three_body_first(num_derivatives_max=6):
    f, u0, (t0, _), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, *, t, p):  # noqa: ARG001
        return f(u, *p)

    return autodiff.taylor_mode_unroll(
        vector_field=vf,
        initial_values=(u0,),
        num=num_derivatives_max,
        t=t0,
        parameters=f_args,
    )


def van_der_pol_second(num_derivatives_max=6):
    f, (u0, du0), (t0, _), f_args = diffeqzoo.ivps.van_der_pol()

    def vf(u, du, *, t, p):  # noqa: ARG001
        return f(u, du, *p)

    return autodiff.taylor_mode_unroll(
        vector_field=vf,
        initial_values=(u0, du0),
        num=num_derivatives_max,
        t=t0,
        parameters=f_args,
    )


if __name__ == "__main__":
    # 64-bit precision and the like
    set_environment()

    solution1 = three_body_first()
    jnp.save("./tests/test_taylor/data/three_body_first_solution.npy", solution1)

    solution2 = van_der_pol_second()
    jnp.save("./tests/test_taylor/data/van_der_pol_second_solution.npy", solution2)

    print("Saving successful.")
