from probdiffeq import taylor
from probdiffeq.backend import flow, func, linalg, np, testing


def test_root_of_jet(num=3):

    def vf(y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = [np.asarray([1.0, 0.0, 0.0])]
    expected = taylor.odejet_unroll(vf, y0, num=num)

    def root(u, du, /):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * u[0] + k3 * u[1] * u[2]
        f1 = k1 * u[0] - k2 * u[1] ** 2 - k3 * u[1] * u[2]

        F1 = du[0] - f0
        F2 = du[1] - f1
        F3 = u[0] + u[1] + u[2] - 1
        # The below is technically not a part of the DAE definition
        # but it is a conserved quantity and providing it really
        # helps the consistent initialisation.
        F4 = du[0] + du[1] + du[2]
        return np.stack([F1, F2, F3, F4])

    is_free = [np.asarray([False, False, False])]
    received = taylor.rootjet_nonlinear_lstsq(
        root, y0, num=num, is_free=is_free, nonlinear_lstsq=levenberg_marquardt
    )

    assert testing.allclose(received, expected)


def levenberg_marquardt(residual, x0, *, num=100):

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
