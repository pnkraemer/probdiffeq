"""Simulate DAEs.

Solve a differential-algebraic equation, namely, the Robertson problem.
The Robertson problem is interesting for many reasons:
  - It comes in DAE, and ODE form
    so we can compare different information operators
  - It has an exponential timescale so (good) adaptive
    steps are needed; fixed steps are hopeless.
  - Its y-states have wildly different scales,
    so a good prior model is important.


"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.special

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(t0=1e-6, t1=1e5) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    @probdiffeq.residual_position_velocity
    def differential(u, du, /, *, t):
        del t
        return du[:2] - dynamics(u)

    def dynamics(y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        return jnp.stack([f0, f1])

    @probdiffeq.residual_position
    def algebraic(u, *, t):
        del t
        return u[0] + u[1] + u[2] - 1

    ssm = probdiffeq.state_space_model_dense()

    jetexpand = probdiffeq.jetexpand_residual(num=4)
    residual = probdiffeq.residual_from_stack(differential, algebraic)
    tcoeffs, _ = jetexpand(residual, [jnp.array([1.0, 0.0, 0.0])], t=t0)

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    base_scale = jnp.asarray([0.8, 2e-05, 0.2])
    init, ioup = ssm.prior_wiener_integrated(tcoeffs, output_scale=base_scale)

    # We build a Jet constraint. Iteration is key, because DAEs are proper stiff.
    linearization_point = probdiffeq.linearization_point_maximum_a_posteriori()
    jet = ssm.constraint_residual(residual, linearization_point=linearization_point)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver_dynamic(strategy=strategy, prior=ioup, constraint=jet)

    # The state-error-estimate doesn't care about the dimension
    # of the DAE, which is exactly what we need here
    error = probdiffeq.error_state_std(constraint=jet, prior=ioup)

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=200)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution = jax.jit(solve)(init, save_at=save_at, atol=1e-9, rtol=1e-7)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5, 5), sharex=True)
    ax[0][0].set_title("Robertson solution", fontsize="medium")
    ax[0][1].set_title("Standard deviations", fontsize="medium")

    # Plot a special index
    i = 0
    ax[0][0].semilogx(save_at, solution.u.mean[i][:, 0] / scipy.special.factorial(i))
    ax[1][0].semilogx(save_at, solution.u.mean[i][:, 1] / scipy.special.factorial(i))
    ax[2][0].semilogx(save_at, solution.u.mean[i][:, 2] / scipy.special.factorial(i))

    ax[0][1].loglog(save_at, solution.u.std[i][:, 0] / scipy.special.factorial(i))
    ax[1][1].loglog(save_at, solution.u.std[i][:, 1] / scipy.special.factorial(i))
    ax[2][1].loglog(save_at, solution.u.std[i][:, 2] / scipy.special.factorial(i))

    ax[0][0].set_ylabel("State $y_1$", fontsize="medium")
    ax[1][0].set_ylabel("State $y_2$", fontsize="medium")
    ax[2][0].set_ylabel("State $y_3$", fontsize="medium")
    ax[2][0].set_xlabel("Time $t$", fontsize="medium")
    ax[2][1].set_xlabel("Time $t$", fontsize="medium")
    ax[0][0].set_xlim((t0, t1))

    plt.tight_layout()
    fig.align_ylabels()
    plt.show()


if __name__ == "__main__":
    main()
