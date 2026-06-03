"""Precompute and save reference solutions. Accelerate testing."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, ode


def three_body_first(num_derivatives_max=10):
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()
    vf = probdiffeq.ode_function(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives_max)
    return jetexpand(vf, (u0,), t=t0)


def van_der_pol_second(num_derivatives_max=10):
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()
    vf = probdiffeq.ode_function(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives_max)
    return jetexpand(vf, (u0, du0), t=t0)


if __name__ == "__main__":
    solution1 = three_body_first()
    np.save(
        "./tests/test_probdiffeq/test_jetexpand/data/three_body_first_solution.npy",
        solution1,
    )

    solution2 = van_der_pol_second()
    np.save(
        "./tests/test_probdiffeq/test_jetexpand/data/van_der_pol_second_solution.npy",
        solution2,
    )
