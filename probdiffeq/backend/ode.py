"""ODE stuff."""

import jax
import jax.experimental.ode


def odeint_and_save_at(vf, y0: tuple, /, save_at, *, atol, rtol):
    assert isinstance(y0, tuple | list)
    assert len(y0) == 1

    def vf_wrapped(y, t):
        return vf(y, t=t)

    return jax.experimental.ode.odeint(vf_wrapped, *y0, save_at, atol=atol, rtol=rtol)


def ivp_lotka_volterra():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, (t0, _), f_args = ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    # Dictionary to ensure pytree compatibility
    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return {"u": f(x["u"], *f_args)}

    return vf, ({"u": u0},), (t0, t1)


def ivp_three_body_1st():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()

    # Dictionary to ensure pytree compatibility
    def vf(u, *, t):  # noqa: ARG001
        return {"u": f(u["u"], *f_args)}

    return vf, ({"u": u0},), (t0, t1)


def ivp_van_der_pol_2nd():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, (u0, du0), (t0, t1), f_args = ivps.van_der_pol()

    def vf(u, du, *, t):  # noqa: ARG001
        return {"u": f(u["u"], du["u"], *f_args)}

    return vf, ({"u": u0}, {"u": du0}), (t0, t1)
