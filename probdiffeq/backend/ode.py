"""ODE stuff."""
import jax.experimental.ode


def odeint(vf, y0, ts, /, *, atol, rtol):
    return jax.experimental.ode.odeint(vf, y0, ts, atol=atol, rtol=rtol)
