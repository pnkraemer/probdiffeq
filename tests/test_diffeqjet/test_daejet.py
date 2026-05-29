"""Tests for the DAE initialization routines."""

from probdiffeq import diffeqjet, probdiffeq
from probdiffeq.backend import func, np, testing


@testing.parametrize("num", [0, 1, 10])
def test_daejet_matches_expectation_on_sir_model(num):

    # Use SIR model because it is structurally similar to DAEs,
    # but really not that hard to solve so we can test in single precision
    # whereas robertson would require double
    def vf_ode(y):
        beta, gamma = 2.0, 0.5  # infection and recovery rates
        S, I, _R = y  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I
        f2 = gamma * I

        return np.stack([f0, f1, f2])

    def algebraic(u, /):
        N = 1.0  # total population
        return u[0] + u[1] + u[2] - N

    def differential(u, du, /):
        beta, gamma = 2.0, 0.5
        S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I

        F1 = du[0] - f0
        F2 = du[1] - f1

        return np.stack([F1, F2])

    y0 = [np.asarray([0.99, 0.01, 0.0])]
    expected = diffeqjet.odejet_unroll(vf_ode, y0, num=num)

    eps = np.finfo_eps(y0[0].dtype)
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=10, tol=eps)

    @func.jit
    def initialize(inits):
        tcoeffs, _info = diffeqjet.daejet_nlstsq(
            differential, algebraic, inits, num=num, nlstsq=nlstsq
        )
        return tcoeffs

    received = initialize(y0)
    assert testing.allclose(received, expected)


@testing.parametrize("num_strides", [0, 1, 3])
@testing.parametrize("stride", [4])
def test_daejet_recursive_matches_expectation_on_sir_model(num_strides, stride):

    # Use SIR model because it is structurally similar to DAEs,
    # but really not that hard to solve so we can test in single precision
    # whereas robertson would require double
    def vf_ode(y):
        beta, gamma = 2.0, 0.5  # infection and recovery rates
        S, I, _R = y  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I
        f2 = gamma * I

        return np.stack([f0, f1, f2])

    def algebraic(u, /):
        N = 1.0  # total population
        return u[0] + u[1] + u[2] - N

    def differential(u, du, /):
        beta, gamma = 2.0, 0.5
        S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I

        F1 = du[0] - f0
        F2 = du[1] - f1

        return np.stack([F1, F2])

    y0 = [np.asarray([0.99, 0.01, 0.0])]
    expected = diffeqjet.odejet_unroll(vf_ode, y0, num=num_strides * stride)
    eps = np.finfo_eps(y0[0].dtype)
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=3, tol=eps)

    @func.jit
    def initialize(inits):
        tcoeffs, _info = diffeqjet.daejet_nlstsq_recursive(
            differential,
            algebraic,
            inits,
            num_strides=num_strides,
            stride=stride,
            nlstsq=nlstsq,
        )
        return tcoeffs

    received = initialize(y0)
    assert testing.allclose(received, expected)
