"""ODE stuff."""
import jax.experimental.ode


def odeint_and_save_at(vf, y0: tuple, /, save_at, *, atol, rtol):
    def vf_wrapped(y, t):
        return vf(y, t=t)

    assert isinstance(y0, (tuple, list))
    assert len(y0) == 1
    return jax.experimental.ode.odeint(vf_wrapped, *y0, save_at, atol=atol, rtol=rtol)
