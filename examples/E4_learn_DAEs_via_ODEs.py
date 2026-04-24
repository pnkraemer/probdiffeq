"""Learn a DAE via learning an ODE."""

import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import diffeqjet, ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# Set up all the configs
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


def main(t0=1e-6, t1=1e2, num_data=100) -> None:
    """Run the script."""

    def vf(y, *, t):
        del t
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

    def while_loop(cond, body, init):
        return jax.lax.while_loop(cond, body, init_val=init)

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    output_scale = jnp.asarray([0.8, 2e-05, 0.2])
    trafo = Trafo(output_scale)

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=num_data)
    solve = solver_fixed(vf, trafo=trafo)

    # True condition
    # key = jax.random.PRNGKey(seed)
    # p_true = 10 * jax.random.uniform(key, shape=(2,)) - 5.0

    p_true = trafo.observed_to_latent(jnp.array([1.0, 0.0, 0.0]))

    # Create data
    solution_true = solve(p_true, save_at=save_at, output_scale=output_scale)
    inputs = solution_true.t
    labels = solution_true.u.mean[0]
    plt.semilogx(inputs, labels * jnp.asarray([[1.0, 1e5, 1.0]]))
    plt.show()
    print()
    print()
    print()
    print()

    @functools.partial(jax.jacfwd, has_aux=True)
    def target(p):
        p_ = jnp.asarray([p, p_true[1]])
        Z1 = solve(p_, output_scale=output_scale, save_at=save_at)
        return Z1.u.std[0][:, 0], (Z1.t, Z1.u.std[0][:, 0])

    Js, (ts, ms) = target(p_true[0] - 10)
    plt.semilogy(ts, jnp.abs(Js))
    plt.show()
    for t, s, J in zip(ts, ms, Js):
        print("t =", t)
        print("s =", s)
        print("J =", J)
        print()

    # TODO: What if we solve the sensitivity ODE?
    # TODO: Diffrax doesn't struggle. So somewhere, we are dividing by zero?
    print()


def solver_fixed(vf, trafo):
    """Create a reverse-mode differentiable probabilistic solver."""

    def solve(p_sqrt, save_at, output_scale):

        y0 = trafo.latent_to_observed(p_sqrt)
        t0, t1 = save_at[0], save_at[-1]

        def vf_auto(u):
            return vf(u, t=t0)

        y0 = diffeqjet.odejet_unroll(vf_auto, [y0], num=2)
        init, ssm = probdiffeq.ssm_taylor(y0)

        prior = probdiffeq.prior_wiener_integrated(ssm=ssm, output_scale=output_scale)

        # We build a Jet constraint. Iteration is key, because DAEs are proper stiff.
        jet = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
        strategy = probdiffeq.strategy_filter(ssm=ssm)
        solver = probdiffeq.solver_dynamic(
            strategy=strategy, prior=prior, constraint=jet, ssm=ssm
        )

        solve = ivpsolve.solve_fixed_grid(solver=solver)
        return solve(init, grid=save_at)

    return solve


if __name__ == "__main__":
    main()
