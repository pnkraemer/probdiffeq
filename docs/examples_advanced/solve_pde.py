# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Simulate a partial differential equation
#

# +
"""Solve a partial differential equation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, ivpsolvers, taylor

jax.config.update("jax_enable_x64", True)


def main():
    """Simulate a PDE."""
    key = jax.random.PRNGKey(1)
    f, (u0,), (t0, t1) = fhn_2d(key, dx=0.025, t1=10.0)

    @jax.jit
    def vf(y, *, t):  # noqa: ARG001
        """Evaluate the dynamics of the PDE."""
        return f(y)

    print("Problem dimension:", u0.size)

    # Set up a state-space model
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="blockdiag")

    # Build a solver
    ts = ivpsolvers.correction_ts1(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_fixedpoint(ssm=ssm)
    solver = ivpsolvers.solver_dynamic(
        ssm=ssm, strategy=strategy, prior=ibm, correction=ts
    )
    adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm)

    # Solve the ODE
    save_at = jnp.linspace(t0, t1, num=5, endpoint=True)
    simulate = simulator(save_at=save_at, adaptive_solver=adaptive_solver, ssm=ssm)
    (u, u_std) = simulate(init)

    fig, axes = plt.subplots(
        nrows=2, ncols=len(u), figsize=(2 * len(u), 3), tight_layout=True
    )
    for t_i, u_i, std_i, ax_i in zip(save_at, u, u_std, axes.T):
        ax_i[0].set_title(f"t = {t_i:.1f}")
        img = ax_i[0].imshow(u_i[0], cmap="copper", vmin=-1, vmax=1)
        plt.colorbar(img)

        uncertainty = jnp.log10(jnp.abs(std_i[0]) + 1e-10)
        img = ax_i[1].imshow(uncertainty, cmap="bone", vmin=-7, vmax=-3)
        plt.colorbar(img)

        ax_i[0].set_xticks(())
        ax_i[1].set_xticks(())
        ax_i[0].set_yticks(())
        ax_i[1].set_yticks(())

    axes[0][0].set_ylabel("PDE solution")
    axes[1][0].set_ylabel("log(stdev)")
    plt.show()


def simulator(save_at, adaptive_solver, ssm):
    """Simulate a PDE."""

    @jax.jit
    def solve(init):
        solution = ivpsolve.solve_adaptive_save_at(
            init, save_at=save_at, dt0=0.1, adaptive_solver=adaptive_solver, ssm=ssm
        )
        return (solution.u[0], solution.u_std[0])

    return solve


def fhn_2d(prng_key, *, dx, t1, t0=0.0, a=2.8e-4, b=5e-3, k=-0.005, tau=1.0):
    """Construct the FitzHugh-Nagumo PDE.

    Source: https://github.com/pnkraemer/tornadox/blob/main/tornadox/ivp.py

    But simplified since Probdiffeq can handle matrix-valued ODEs.
    Here, we also set tau = 1.0 to make the example quick to execute.
    """
    ny, nx = int(1 / dx), int(1 / dx)

    y0 = jax.random.uniform(prng_key, shape=(2, ny, nx))

    @jax.jit
    def fhn_2d(x):
        u, v = x
        du = _laplace_2d(u, dx=dx)
        dv = _laplace_2d(v, dx=dx)
        u_new = a * du + u - u**3 - v + k
        v_new = (b * dv + u - v) / tau
        return jnp.stack((u_new, v_new))

    return fhn_2d, (y0,), (t0, t1)


def _laplace_2d(grid, dx):
    """2D Laplace operator on a vectorized 2d grid."""
    # Set the boundary values to the nearest interior node
    # This enforces Neumann conditions.
    padded_grid = jnp.pad(grid, pad_width=1, mode="edge")

    # Laplacian via convolve2d()
    kernel = jnp.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel /= dx**2
    grid = jax.scipy.signal.convolve2d(padded_grid, kernel, mode="same")
    return grid[1:-1, 1:-1]


if __name__ == "__main__":
    main()
