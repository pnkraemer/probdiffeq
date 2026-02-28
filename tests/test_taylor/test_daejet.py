from probdiffeq import taylor
from probdiffeq.backend import flow, func, linalg, np, testing


def test_daejet_matches_expectation_on_sir_model(num=3):

    # Use SIR model because it is structurally similar to DAEs,
    # but really not that hard to solve so we can test in single precision
    # whereas robertson would require double
    def vf_ode(y):
        beta, gamma = 2.0, 0.5  # infection and recovery rates
        S, I, R = y

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I
        f2 = gamma * I

        return np.stack([f0, f1, f2])

    def algebraic(u, /):
        N = 1.0  # total population
        return u[0] + u[1] + u[2] - N

    def differential(u, du, /):
        beta, gamma = 2.0, 0.5
        S, I, _R = u

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I

        F1 = du[0] - f0
        F2 = du[1] - f1

        return np.stack([F1, F2])

    y0 = [np.asarray([0.99, 0.01, 0.0])]
    expected = taylor.odejet_unroll(vf_ode, y0, num=num)

    received = taylor.daejet_nonlinear_lstsq(
        differential, algebraic, y0, num=num, nonlinear_lstsq=levenberg_marquardt
    )
    assert testing.allclose(received, expected)


def levenberg_marquardt(residual, x0, *, num=1000):

    # Damping equivalent to machine epsilon (small seems desirable here)
    damping = 10 * np.finfo_eps(x0.dtype)

    def body_fun(x, _):
        Fx = residual(x)
        J = func.jacfwd(residual)(x)
        A = J.T @ J + damping * np.eye(x.shape[0])
        b = -J.T @ Fx
        dx = linalg.solve_lu(A, b)
        return x + dx, None

    x, _ = flow.scan(body_fun, x0, xs=None, length=num)

    return x
