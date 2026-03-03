"""Tests for the DAE initialization routines."""

from probdiffeq import taylor
from probdiffeq.backend import flow, func, linalg, np, structs, testing, tree
from probdiffeq.backend.typing import Array


@testing.parametrize("num", [2, 4, 10])
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
    expected = taylor.odejet_unroll(vf_ode, y0, num=num)

    lstsq = levenberg_marquardt(maxiter=5)
    received, _info = taylor.daejet_nonlinear_lstsq(
        differential, algebraic, y0, num=num, nonlinear_lstsq=lstsq
    )
    assert testing.allclose(received, expected)


def levenberg_marquardt(*, maxiter):
    def solve(residual, x0):
        # TODO: include DAEJET in a Taylor series benchmark
        # (I really wonder whether it is faster or slower than rewrite+odejet)

        @tree.register_dataclass
        @structs.dataclass
        class State:
            x: Array
            fx: Array
            i: int

        # Damping equivalent to machine epsilon
        # (small seems desirable here but what do I know...)
        damping = 10 * np.finfo_eps(x0.dtype)

        def cond_fun(state: State):
            cond1 = linalg.vector_norm(state.fx) > damping
            cond2 = state.i < maxiter
            return np.logical_and(cond1, cond2)

        def body_fun(state: State) -> State:
            J = func.jacfwd(residual)(state.x)
            A = J.T @ J + damping * np.eye(state.x.shape[0])
            b = -J.T @ state.fx
            dx = linalg.solve_lu(A, b)

            xnew = state.x + dx
            fxnew = residual(xnew)
            return State(xnew, fxnew, i=state.i + 1)

        init = State(x0, residual(x0), i=0)
        final = flow.while_loop(cond_fun, body_fun, init=init)
        return final.x, {"iters": final.i}

    return solve
