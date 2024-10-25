"""Test the controllers."""

from probdiffeq import ivpsolve
from probdiffeq.backend import numpy as np


def test_equivalence_pi_vs_i(dt=0.1428, norm=3.142, rate=3, num_applies=4):
    ctrl_pi = ivpsolve.control_proportional_integral(
        power_integral_unscaled=1.0, power_proportional_unscaled=0.0
    )
    ctrl_i = ivpsolve.control_integral()

    x_pi = ctrl_pi.init(dt)
    for _ in range(num_applies):
        x_pi = ctrl_pi.apply(x_pi, norm, rate)
    x_pi = ctrl_pi.extract(x_pi)

    x_i = ctrl_i.init(dt)
    for _ in range(num_applies):
        x_i = ctrl_i.apply(x_i, norm, rate)
    x_i = ctrl_i.extract(x_i)
    assert np.allclose(x_i, x_pi)
