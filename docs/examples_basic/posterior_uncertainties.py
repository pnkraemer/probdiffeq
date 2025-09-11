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

# # Posterior uncertainties

# +
"""Display the marginal uncertainties of filters and smoothers."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor

# Set up the ODE


def vf(y, *, t):  # noqa: ARG001
    """Evaluate the Lotka-Volterra vector field."""
    y0, y1 = y[0], y[1]

    y0_new = 0.5 * y0 - 0.05 * y0 * y1
    y1_new = -0.5 * y1 + 0.05 * y0 * y1
    return jnp.asarray([y0_new, y1_new])


t0 = 0.0
t1 = 2.0
u0 = jnp.asarray([20.0, 20.0])


# Set up a solver
# To all users: Try replacing the fixedpoint-smoother with a filter!
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=3)
init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="isotropic")
ts = ivpsolvers.correction_ts1(vf, ssm=ssm)
strategy = ivpsolvers.strategy_fixedpoint(ssm=ssm)
solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts, ssm=ssm)
adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-1, rtol=1e-1, ssm=ssm)

# Solve the ODE
ts = jnp.linspace(t0, t1, endpoint=True, num=50)
sol = ivpsolve.solve_adaptive_save_at(
    init, save_at=ts, dt0=0.1, adaptive_solver=adaptive_solver, ssm=ssm
)

# Calibrate
marginals = stats.calibrate(sol.marginals, output_scale=sol.output_scale, ssm=ssm)
std = ssm.stats.standard_deviation(marginals)
u_std = ssm.stats.qoi_from_sample(std)

# Plot the solution
fig, axes = plt.subplots(
    nrows=3,
    ncols=len(tcoeffs),
    sharex="col",
    tight_layout=True,
    figsize=(len(u_std) * 2, 5),
)
for i, (u_i, std_i, ax_i) in enumerate(zip(sol.u, u_std, axes.T)):
    # Set up titles and axis descriptions
    if i == 0:
        ax_i[0].set_title("State")
        ax_i[0].set_ylabel("Prey")
        ax_i[1].set_ylabel("Predators")
        ax_i[2].set_ylabel("Std.-dev.")
    elif i == 1:
        ax_i[0].set_title(f"{i}st deriv.")
    elif i == 2:
        ax_i[0].set_title(f"{i}nd deriv.")
    elif i == 3:
        ax_i[0].set_title(f"{i}rd deriv.")
    else:
        ax_i[0].set_title(f"{i}th deriv.")

    ax_i[-1].set_xlabel("Time")

    for m, std, ax in zip(u_i.T, std_i.T, ax_i):
        # Plot the mean
        ax.plot(sol.t, m)

        # Plot the standard deviation
        lower, upper = m - 1.96 * std, m + 1.96 * std
        ax.fill_between(sol.t, lower, upper, alpha=0.3)
        ax.set_xlim((jnp.amin(ts), jnp.amax(ts)))

    ax_i[2].semilogy(sol.t, std_i[:, 0], label="Prey")
    ax_i[2].semilogy(sol.t, std_i[:, 1], label="Predators")
    ax_i[2].legend(fontsize="x-small")

fig.align_ylabels()
plt.show()
