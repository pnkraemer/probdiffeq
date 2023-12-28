"""ODE stuff."""
import jax.experimental.ode


def odeint_and_save_at(vf, y0, /, save_at, *, atol, rtol):
    return jax.experimental.ode.odeint(vf, y0, save_at, atol=atol, rtol=rtol)
