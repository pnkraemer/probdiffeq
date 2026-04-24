import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class Trafo:
    """Coordinate transformation to make the optimisation problem well-posed."""

    def __init__(self, scale):
        # e.g. jnp.array([1., 1e-5, 1e-3])
        self.scale = jnp.array(scale)

    def observed_to_latent(self, x, eps=1e-6):
        """Simplex R^3 -> unconstrained R^2."""
        x = x / self.scale
        x = jnp.clip(x, eps, 1.0 - eps)
        x = x / x.sum()  # renormalise
        return jnp.log(x[:-1] / x[-1])

    def latent_to_observed(self, u):
        """Unconstrained R^2 -> simplex R^3."""
        u_full = jnp.append(u, 0.0)
        u_full = u_full - jnp.max(u_full)
        e = jnp.exp(u_full)
        x = e / e.sum()

        # Rescale back
        x *= self.scale
        return x / x.sum()


class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])


def main(p_latent):
    trafo = Trafo(jnp.asarray([0.8, 2e-05, 0.2]))
    robertson = Robertson(0.04, 3e7, 1e4)
    terms = diffrax.ODETerm(robertson)

    t0 = 1e-6
    t1 = 1e5

    y0 = trafo.latent_to_observed(p_latent)
    dt0 = 0.0002
    solver = diffrax.Kvaerno5()

    save_at = jnp.asarray(
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    )

    saveat = diffrax.SaveAt(ts=save_at)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=diffrax.ForwardMode(),
    )
    return sol


def fun(p0):
    sol = main(p0)
    # return sol.ys / jnp.asarray([0.8, 2e-05, 0.2])[None, :], sol.ts
    return sol.ys / jnp.asarray([1.0, 1e-4, 1.0])[None, :], sol.ts


if __name__ == "__main__":
    trafo = Trafo(jnp.asarray([0.8, 2e-05, 0.2]))
    p = trafo.observed_to_latent(jnp.array([1.0, 0.0, 0.0]))

    y, t = fun(p)
    plt.semilogx(t, y)
    plt.show()
    Js, ts = jax.jit(jax.jacfwd(fun, has_aux=True))(p)
    for t, J in zip(ts, Js):
        print(t)
        print(J)
        print()
