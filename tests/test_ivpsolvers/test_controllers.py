"""Test the controllers."""

from probdiffeq import ivpsolvers
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing


@testing.parametrize("dt", [0.1428])
@testing.parametrize("error_power", [3.142])
@testing.parametrize("num_applies", [4])
def test_equivalence_pi_vs_i(dt, error_power, num_applies):
    ctrl_pi = ivpsolvers.control_proportional_integral(
        power_integral_unscaled=1.0, power_proportional_unscaled=0.0
    )
    ctrl_i = ivpsolvers.control_integral()

    x_pi = ctrl_pi.init(dt)
    dt_pi = dt
    for _ in range(num_applies):
        dt_pi, x_pi = ctrl_pi.apply(dt_pi, x_pi, error_power=error_power)

    x_i = ctrl_i.init(dt)
    dt_i = dt
    for _ in range(num_applies):
        dt_i, x_i = ctrl_i.apply(dt_i, x_i, error_power=error_power)
    assert np.allclose(dt_i, dt_pi)
