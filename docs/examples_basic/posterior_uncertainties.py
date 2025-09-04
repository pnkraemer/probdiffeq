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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.util.doc_util import notebook

# -

plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())

# +
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# -

# Set an example problem.
#
# Solve the problem on a low resolution and
# short time-span to achieve large uncertainty.

# +
f, u0, (t0, t1), f_args = ivps.lotka_volterra()


@jax.jit
def vf(*ys, t):  # noqa: ARG001
    """Evaluate the Lotka-Volterra vector field."""
    return f(*ys, *f_args)


# -

# ## Filter

# +
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, output_scale=1.0, ssm_fact="isotropic")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_filter(ssm=ssm)
solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

ts = jnp.linspace(t0, t0 + 2.0, endpoint=True, num=500)

# +
dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), (u0,))

init = solver.initial_condition()
sol = ivpsolve.solve_adaptive_save_at(
    vf, init, save_at=ts, dt0=dt0, adaptive_solver=adaptive_solver, ssm=ssm
)

marginals = stats.calibrate(sol.marginals, output_scale=sol.output_scale, ssm=ssm)
# -

# Plot the solution

# +
_, num_derivatives, _ = marginals.mean.shape


fig, axes_all = plt.subplots(
    nrows=2, ncols=num_derivatives, sharex=True, tight_layout=True, figsize=(8, 3)
)

for i, axes_cols in enumerate(axes_all.T):
    ms = marginals.mean[:, i, :]
    ls = marginals.cholesky[:, i, :]
    stds = jnp.sqrt(jnp.einsum("jn,jn->j", ls, ls))

    if i == 1:
        axes_cols[0].set_title(f"{i}st deriv.")
    elif i == 2:
        axes_cols[0].set_title(f"{i}nd deriv.")
    elif i == 3:
        axes_cols[0].set_title(f"{i}rd deriv.")
    else:
        axes_cols[0].set_title(f"{i}th deriv.")

    axes_cols[0].plot(sol.t, ms, marker="None")
    for m in ms.T:
        axes_cols[0].fill_between(sol.t, m - 1.96 * stds, m + 1.96 * stds, alpha=0.3)

    axes_cols[1].semilogy(sol.t, stds, marker="None")

plt.show()
# -

# ## Smoother

# +
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, output_scale=1.0, ssm_fact="isotropic")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_fixedpoint(ssm=ssm)
solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

ts = jnp.linspace(t0, t0 + 2.0, endpoint=True, num=500)

# +
init = solver.initial_condition()
sol = ivpsolve.solve_adaptive_save_at(
    vf, init, save_at=ts, dt0=dt0, adaptive_solver=adaptive_solver, ssm=ssm
)

marginals = stats.calibrate(sol.marginals, output_scale=sol.output_scale, ssm=ssm)
posterior = stats.calibrate(sol.posterior, output_scale=sol.output_scale, ssm=ssm)
posterior = stats.markov_select_terminal(posterior)
# -

key = jax.random.PRNGKey(seed=1)
samples, _init = stats.markov_sample(key, posterior, shape=(2,), reverse=True, ssm=ssm)

# +
_, num_derivatives, _ = marginals.mean.shape


fig, axes_all = plt.subplots(
    nrows=2, ncols=num_derivatives, sharex=True, tight_layout=True, figsize=(8, 3)
)

for i, axes_cols in enumerate(axes_all.T):
    samps = samples[i]
    ms = ssm.stats.qoi_from_sample(marginals.mean)[i]
    ls = ssm.stats.qoi_from_sample(marginals.cholesky)[i]
    stds = jnp.sqrt(jnp.einsum("jn,jn->j", ls, ls))

    if i == 1:
        axes_cols[0].set_title(f"{i}st deriv.")
    elif i == 2:
        axes_cols[0].set_title(f"{i}nd deriv.")
    elif i == 3:
        axes_cols[0].set_title(f"{i}rd deriv.")
    else:
        axes_cols[0].set_title(f"{i}th deriv.")

    axes_cols[0].plot(sol.t, ms, marker="None")
    for s in samps:
        axes_cols[0].plot(
            sol.t[:-1], s[..., 0], color="C0", linewidth=0.35, marker="None"
        )
        axes_cols[0].plot(
            sol.t[:-1], s[..., 1], color="C1", linewidth=0.35, marker="None"
        )
    for m in ms.T:
        axes_cols[0].fill_between(sol.t, m - 1.96 * stds, m + 1.96 * stds, alpha=0.3)

    axes_cols[1].semilogy(sol.t, stds, marker="None")

plt.show()
# -

# The marginal standard deviations (bottom row)
# show how the filter is forward-only,
# whereas the smoother is a global estimate.
#
# This is why you should use a filter for
# terminal-value simulation and a smoother if you want "global" solutions.
