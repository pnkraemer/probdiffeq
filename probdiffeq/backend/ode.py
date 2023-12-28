"""ODE stuff."""
import diffrax
import jax
import jax.experimental.ode
import jax.numpy as jnp


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
