"""ODE stuff."""

from collections import namedtuple

import jax
import jax.experimental.ode
import jax.numpy as jnp


def odeint_and_save_at(vf, y0: tuple, /, save_at, *, atol, rtol):
    assert isinstance(y0, (tuple, list))
    assert len(y0) == 1

    save_at = jnp.asarray(save_at)

    sign = jnp.sign(jnp.diff(save_at[:2])[0])  # +1 forward, -1 reversed

    def vf_wrapped(y, t):
        # forward (sign=+1): s = t,  returns vf(y, t=s)
        # reversed (sign=-1): s = -t, returns -vf(y, t=-s)
        vfx = vf(y, t=sign * t)
        return jax.tree_util.tree_map(lambda s: sign * s, vfx)

    return jax.experimental.ode.odeint(
        vf_wrapped, *y0, save_at[:: int(sign)], atol=atol, rtol=rtol
    )


def ivp_lotka_volterra():
    t1 = 2.0  # Short time-intervals are sufficient for this test.
    t0, t1 = (0.0, 2.0)

    # Use a crazy pytree structure to build the ODE

    PredPrey = namedtuple("PredPrey", ["predators", "prey"])
    u0 = {"U": PredPrey(predators=jnp.asarray([[[20.0]]]), prey=jnp.asarray(20.0))}

    @jax.jit
    def vf(x: dict, /, *, t) -> dict:  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        y0, y1 = f((x["U"].predators.squeeze(), x["U"].prey.squeeze()))
        return {"U": PredPrey(predators=y0.reshape((1, 1, 1)), prey=y1.reshape(()))}

    def f(y, /):
        a, b, c, d = 0.5, 0.05, 0.5, 0.05
        y0_y1_a = y[0] * y[1]
        y0_y1_b = y[0] * y[1]
        return [a * y[0] - b * y0_y1_a, -c * y[1] + d * y0_y1_b]

    return vf, (u0,), (t0, t1)


def ivp_three_body_1st():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()

    # Dictionary to ensure pytree compatibility
    def vf(u, /, *, t):
        del t
        return {"u": f(u["u"], *f_args)}

    return vf, ({"u": u0},), (t0, t1)


def ivp_van_der_pol_2nd():
    # Local imports because diffeqzoo is not an official dependency
    from diffeqzoo import backend, ivps

    if not backend.has_been_selected:
        backend.select("jax")

    f, (u0, du0), (t0, t1), f_args = ivps.van_der_pol()

    def vf(u, du, /, *, t):  # noqa: ARG001
        return {"u": f(u["u"], du["u"], *f_args)}

    return vf, ({"u": u0}, {"u": du0}), (t0, t1)
