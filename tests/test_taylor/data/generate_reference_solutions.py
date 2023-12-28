"""Precompute and save reference solutions. Accelerate testing."""

from probdiffeq.backend import config, ode
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode
from probdiffeq.taylor import autodiff


def set_environment():
    """Set the environment (e.g., 64-bit precision).

    The setup used to precompute references should match that of the other tests.
    """
    # Test on CPU.
    config.update("platform_name", "cpu")

    # Double precision
    config.update("enable_x64", True)


def three_body_first(num_derivatives_max=6):
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()

    return autodiff.taylor_mode_unroll(
        lambda y: vf(y, t=t0), (u0,), num=num_derivatives_max
    )


def van_der_pol_second(num_derivatives_max=6):
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()

    return autodiff.taylor_mode_unroll(
        lambda *ys: vf(*ys, t=t0), (u0, du0), num=num_derivatives_max
    )


if __name__ == "__main__":
    # 64-bit precision and the like
    set_environment()

    solution1 = three_body_first()
    np.save("./tests/test_taylor/data/three_body_first_solution.npy", solution1)

    solution2 = van_der_pol_second()
    np.save("./tests/test_taylor/data/van_der_pol_second_solution.npy", solution2)

    print("Saving successful.")
