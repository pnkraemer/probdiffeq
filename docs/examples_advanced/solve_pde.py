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

# # Quickstart
#
# Let's have a look at an easy example.

# +
"""Solve a partial differential equation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, ivpsolvers


def main():
    """Simulate a PDE."""
    key = jax.random.PRNGKey(2)
    v = 1.25
    f, (u0,), (t0, t1) = fhn_2d(key, bbox=[[-v, -v], [v, v]], dx=0.05, t1=10.0)

    @jax.jit
    def vf(y, *, t):  # noqa: ARG001
        """Evaluate the dynamics of the logistic ODE."""
        return f(y)

    print(u0.shape)
    print(vf(u0, t=t0).shape)

    # Set up a state-space model
    tcoeffs = [u0, vf(u0, t=t0)]
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="blockdiag")

    # Build a solver
    ts = ivpsolvers.correction_ts1(vf, ssm=ssm, ode_order=1)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_dynamic(
        ssm=ssm, strategy=strategy, prior=ibm, correction=ts
    )
    adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm, atol=1e-3, rtol=1e-3)

    # Solve the ODE
    # To all users: Try different solution routines.
    save_at = jnp.linspace(t0, t1, num=5)
    simulate = simulator(save_at=save_at, adaptive_solver=adaptive_solver, ssm=ssm)
    u, u_std = simulate(init)
    # u.block_until_ready()
    print("ok")
    print(u_std)

    u, u_std = simulate(init)
    # u.block_until_ready()
    print("ok again")
    fig, axes = plt.subplots(
        nrows=2, ncols=len(u), figsize=(2 * len(u), 3), tight_layout=True, dpi=200
    )
    for u_i, std_i, ax_i in zip(u, u_std, axes.T):
        ax_i[0].imshow(u_i[0], vmin=-1, vmax=1, cmap="copper")
        ax_i[1].imshow(jnp.abs(std_i[0]) ** 2, cmap="copper")

    plt.show()


def simulator(save_at, adaptive_solver, ssm):
    """Simulate a PDE."""

    @jax.jit
    def solve(init):
        solution = ivpsolve.solve_adaptive_save_at(
            init, save_at=save_at, dt0=0.1, adaptive_solver=adaptive_solver, ssm=ssm
        )
        return solution.u[0], solution.u_std[0]

    return solve


def fhn_2d(
    prng_key, t0=0.0, t1=20.0, bbox=None, dx=0.02, a=2.8e-4, b=5e-3, k=-0.005, tau=0.1
):
    """Construct the FitzHugh-Nagumo PDE.

    Source: https://github.com/pnkraemer/tornadox/blob/main/tornadox/ivp.py

    (But simplified since Probdiffeq can handle matrix-valued ODEs)
    """
    if bbox is None:
        bbox = [[0.0, 0.0], [1.0, 1.0]]

    ny, nx = int((bbox[1][0] - bbox[0][0]) / dx), int((bbox[1][1] - bbox[0][1]) / dx)

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
    kernel = (
        1 / (dx**2) * jnp.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    )
    grid = jax.scipy.signal.convolve2d(padded_grid, kernel, mode="same")
    return grid[1:-1, 1:-1]


if __name__ == "__main__":
    main()
