"""Test the controllers."""

from probdiffeq import ivpsolvers
from probdiffeq.backend import numpy as np


def test_equivalence_pi_vs_i(dt=0.1428, error_power=3.142, num_applies=4):
    ctrl_pi = ivpsolvers.control_proportional_integral(
        power_integral_unscaled=1.0, power_proportional_unscaled=0.0
    )
    ctrl_i = ivpsolvers.control_integral()

    x_pi = ctrl_pi.init(dt)
    for _ in range(num_applies):
        x_pi = ctrl_pi.apply(x_pi, error_power=error_power)
    x_pi = ctrl_pi.extract(x_pi)

    x_i = ctrl_i.init(dt)
    for _ in range(num_applies):
        x_i = ctrl_i.apply(x_i, error_power=error_power)
    x_i = ctrl_i.extract(x_i)
    assert np.allclose(x_i, x_pi)
