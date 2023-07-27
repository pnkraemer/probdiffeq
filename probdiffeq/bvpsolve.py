"""BVP solver."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace.scalar import corr, extra, linearise, variables


def solve_fixed_grid(vf, bcond, grid, *, ode_shape=(), num_derivatives=2):
    # Prior distribution
    init = variables.standard_normal(num_derivatives + 1)
    extrapolation = extra.ibm_discretise(grid, num_derivatives=num_derivatives)

    # Condition on boundary constraints
    _shape = (num_derivatives + 1,)
    correction_bcond = _correction_model_bcond(bcond, grid, state_shape=_shape)
    extra_bcond = _kalmanfilter_bcond(init, extrapolation, correction_bcond)
    return extra_bcond

    # Condition on ODE constraints
    correction_ode = _correction_model_ode(vf, grid)
    data_ode = _zeros(grid)
    solution = _kalmanfilter_ode(extra_bcond, correction_ode)

    # Return solution
    return solution


def _correction_model_bcond(bcond, grid, *, state_shape):
    g0, g1 = bcond

    unimportant_value = jnp.ones(state_shape)
    A_left = linearise.ts1(lambda x: g0(x[0, ...]), unimportant_value)
    A_right = linearise.ts1(lambda x: g1(x[0, ...]), unimportant_value)
    return A_left, A_right


def _kalmanfilter_bcond(init, extrapolation, correction):
    (a, q_sqrtm), (p, p_inv) = extrapolation

    H_left, H_right = correction

    # Initialise on the right end (we filter in reverse)
    rv = corr.correct_affine_qoi(init, H_right)

    pass


def _correction_model_ode(vf, mesh):
    pass


def _kalmanfilter(extra, corr, data, reverse=False):
    pass
