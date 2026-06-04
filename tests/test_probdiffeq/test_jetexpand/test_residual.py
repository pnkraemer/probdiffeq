"""Tests for the DAE initialization routines."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, testing


@testing.parametrize("num", [0, 1, 10])
def test_residual_init_matches_expectation_on_sir_model(num):
    # todo: rename linearization to meanfinding or so
    # Baseline
    y0 = [np.asarray([0.99, 0.01, 0.0])]
    jetexpand_ode = probdiffeq.jetexpand_ode_unroll(num=num)
    expected, _ = jetexpand_ode(vf_ode, y0, t=0.0)

    # Residual via DAE
    differential_lifted = probdiffeq.jet_lift(differential, lift_by=len(expected) - 2)
    algebraic_lifted = probdiffeq.jet_lift(algebraic, lift_by=len(expected) - 1)
    residual = probdiffeq.residual_from_stack(differential_lifted, algebraic_lifted)

    # Jet expansion
    jetexpand = probdiffeq.jetexpand_residual(num=num)
    jetexpand = func.jit(jetexpand, static_argnums=(0,))
    received, _info = jetexpand(residual, y0, t=0.0)
    assert testing.allclose(received, expected)


@probdiffeq.ode
def vf_ode(y, /, *, t):
    del t
    beta, gamma = 2.0, 0.5  # infection and recovery rates
    S, I, _R = y  # noqa: E741 ("I" is a good variable name in an SIR model)

    f0 = -beta * S * I
    f1 = beta * S * I - gamma * I
    f2 = gamma * I

    return np.stack([f0, f1, f2])


@probdiffeq.residual_state
def algebraic(u, /, *, t):
    del t
    N = 1.0  # total population
    return u[0] + u[1] + u[2] - N


@probdiffeq.residual_state_velocity
def differential(u, du, /, *, t):
    del t
    beta, gamma = 2.0, 0.5
    S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

    f0 = -beta * S * I
    f1 = beta * S * I - gamma * I

    F1 = du[0] - f0
    F2 = du[1] - f1

    return np.stack([F1, F2])
