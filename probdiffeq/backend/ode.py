"""ODE stuff."""
import diffeqzoo.ivps
import diffrax
import jax
import jax.experimental.ode
import jax.numpy as jnp
from diffeqzoo import backend

# ODE examples must be in JAX
backend.select("jax")


def odeint_and_save_at(vf, y0: tuple, /, save_at, *, atol, rtol):
    assert isinstance(y0, (tuple, list))
    assert len(y0) == 1

    def vf_wrapped(y, t):
        return vf(y, t=t)

    return jax.experimental.ode.odeint(vf_wrapped, *y0, save_at, atol=atol, rtol=rtol)


def odeint_dense(vf, y0: tuple, /, t0, t1, *, atol, rtol):
    assert isinstance(y0, (tuple, list))
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


def ivp_logistic():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.logistic()
    t1 = 0.75

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return f(x, *f_args)

    return vf, (u0,), (t0, t1)


def ivp_lotka_volterra():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return f(x, *f_args)

    return vf, (u0,), (t0, t1)


def ivp_affine_multi_dimensional():
    t0, t1 = 0.0, 2.0
    u0 = jnp.ones((2,))

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return 2 * x

    def solution(t):
        return jnp.exp(2 * t) * jnp.ones((2,))

    return vf, (u0,), (t0, t1), solution


def ivp_affine_scalar():
    t0, t1 = 0.0, 2.0
    u0 = 1.0

    @jax.jit
    def vf(x, *, t):  # noqa: ARG001
        return 2 * x

    def solution(t):
        return jnp.exp(2 * t)

    return vf, (u0,), (t0, t1), solution


def ivp_three_body_1st():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, *, t):  # noqa: ARG001
        return f(u, *f_args)

    return vf, (u0,), (t0, t1)


def ivp_van_der_pol_2nd():
    f, (u0, du0), (t0, t1), f_args = diffeqzoo.ivps.van_der_pol()

    def vf(u, du, *, t):  # noqa: ARG001
        return f(u, du, *f_args)

    return vf, (u0, du0), (t0, t1)
