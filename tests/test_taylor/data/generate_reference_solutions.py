"""Precompute and save reference solutions. Accelerate testing."""

from probdiffeq import taylor
from probdiffeq.backend import config, functools, ode
from probdiffeq.backend import numpy as np


def three_body_first(num_derivatives_max=10):
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()
    vf = functools.partial(vf, t=t0)
    return taylor.odejet_unroll(vf, (u0,), num=num_derivatives_max)


def van_der_pol_second(num_derivatives_max=10):
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()
    vf = functools.partial(vf, t=t0)
    return taylor.odejet_unroll(vf, (u0, du0), num=num_derivatives_max)


if __name__ == "__main__":
    # Double precision
    config.update("enable_x64", True)

    solution1 = three_body_first()
    np.save("./tests/test_taylor/data/three_body_first_solution.npy", solution1)

    solution2 = van_der_pol_second()
    np.save("./tests/test_taylor/data/van_der_pol_second_solution.npy", solution2)
