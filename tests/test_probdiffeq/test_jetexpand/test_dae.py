"""Tests for the DAE initialization routines."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, testing


@testing.parametrize("num", [0, 1, 10])
def test_daejet_matches_expectation_on_sir_model(num):

    y0 = [np.asarray([0.99, 0.01, 0.0])]
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=num)
    expected, _ = jetexpand(vf_ode, y0, t=0.0)

    dae = probdiffeq.dae(differential=differential, algebraic=algebraic)
    eps = np.finfo_eps(y0[0].dtype)
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=10, tol=eps)
    jetexpand = probdiffeq.jetexpand_dae_nlstsq(num=num, nlstsq=nlstsq)
    jetexpand = func.jit(jetexpand, static_argnums=(0,))
    received, _info = jetexpand(dae, y0, t=0.0)
    assert testing.allclose(received, expected)


@testing.parametrize("num_strides", [0, 1, 3])
@testing.parametrize("stride", [4])
def test_daejet_recursive_matches_expectation_on_sir_model(num_strides, stride):

    # Use SIR model because it is structurally similar to DAEs,
    # but really not that hard to solve so we can test in single precision
    # whereas robertson would require double

    y0 = [np.asarray([0.99, 0.01, 0.0])]
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_strides * stride)
    expected, _ = jetexpand(vf_ode, y0, t=0.0)

    eps = np.finfo_eps(y0[0].dtype)
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=3, tol=eps)
    dae = probdiffeq.dae(differential=differential, algebraic=algebraic)

    jetexpand = probdiffeq.jetexpand_dae_nlstsq_recursive(
        num_strides=num_strides, stride=stride, nlstsq=nlstsq
    )
    jetexpand = func.jit(jetexpand, static_argnums=0)
    received, _info = jetexpand(dae, y0, t=0.0)
    assert testing.allclose(received, expected)


@probdiffeq.ode_vector_field
def vf_ode(y, /, *, t):
    del t
    beta, gamma = 2.0, 0.5  # infection and recovery rates
    S, I, _R = y  # noqa: E741 ("I" is a good variable name in an SIR model)

    f0 = -beta * S * I
    f1 = beta * S * I - gamma * I
    f2 = gamma * I

    return np.stack([f0, f1, f2])


@probdiffeq.root_state
def algebraic(u, /, *, t):
    del t
    N = 1.0  # total population
    return u[0] + u[1] + u[2] - N


@probdiffeq.root_state_and_velocity
def differential(u, du, /, *, t):
    del t
    beta, gamma = 2.0, 0.5
    S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

    f0 = -beta * S * I
    f1 = beta * S * I - gamma * I

    F1 = du[0] - f0
    F2 = du[1] - f1

    return np.stack([F1, F2])
