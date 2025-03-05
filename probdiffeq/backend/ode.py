"""ODE stuff."""

import jax
import jax.experimental.ode
import jax.numpy as jnp


def odeint_and_save_at(vf, y0: tuple, /, save_at, *, atol, rtol):
    assert isinstance(y0, tuple | list)
    assert len(y0) == 1

    def vf_wrapped(y, t):
        return vf(y, t=t)

    return jax.experimental.ode.odeint(vf_wrapped, *y0, save_at, atol=atol, rtol=rtol)


def odeint_dense(vf, y0: tuple, /, t0, t1, *, atol, rtol):
    # Local import because diffrax is not an official dependency
    import diffrax

    assert isinstance(y0, tuple | list)
    assert len(y0) == 1

    @diffrax.ODETerm
    @jax.jit
    def vf_wrapped(t, y, _args):
        return vf(y, t=t)

    solution_object = diffrax.diffeqsolve(
        vf_wrapped,
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0[0],
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(atol=atol, rtol=rtol),
    )

    def solution(t):
        # Automatic batching
        if jnp.ndim(t) > 0:
            return jax.vmap(solution)(t)

        # Interpolate
        return solution_object.evaluate(t)

    return solution


def ivp_lotka_volterra():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, (t0, _), f_args = ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return f(x, *f_args)

    return vf, (u0,), (t0, t1)


def ivp_three_body_1st():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()

    def vf(u, *, t):  # noqa: ARG001
        return f(u, *f_args)

    return vf, (u0,), (t0, t1)


def ivp_van_der_pol_2nd():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, (u0, du0), (t0, t1), f_args = ivps.van_der_pol()

    def vf(u, du, *, t):  # noqa: ARG001
        return f(u, du, *f_args)

    return vf, (u0, du0), (t0, t1)
