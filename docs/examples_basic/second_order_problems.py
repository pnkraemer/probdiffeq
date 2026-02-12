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

# # Second-order systems

# +
"""Demonstrate how to solve second-order IVPs without transforming them first."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, probdiffeq, taylor

# +
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

# -

# Quick refresher: first-order ODEs

# +
f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()


@jax.jit
def vf_1(y, t):  # noqa: ARG001
    """Evaluate the three-body problem as a first-order IVP."""
    return f(y, *f_args)


tcoeffs = taylor.odejet_padded_scan(lambda y: vf_1(y, t=t0), (u0,), num=4)
init, ibm, ssm = probdiffeq.prior_wiener_integrated(
    tcoeffs, output_scale=1.0, ssm_fact="isotropic"
)
ts0 = probdiffeq.correction_ts0(vf_1, ssm=ssm)
strategy = probdiffeq.strategy_filter(ssm=ssm)
solver_1st = probdiffeq.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
errorest_1st = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)


# -
save_at = jnp.linspace(t0, t1, endpoint=True, num=250)
solve = ivpsolve.solve_adaptive_save_at(solver=solver_1st, errorest=errorest_1st)
solution = jax.jit(solve)(init, save_at=save_at, atol=1e-5, rtol=1e-5)
norm = jnp.linalg.norm((solution.u.mean[0][-1] - u0) / jnp.abs(1.0 + u0))
plt.title(f"error={norm:.3f}")
plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
plt.show()

# The default configuration assumes that the ODE to be solved is of first order.
# Now, the same game with a second-order ODE

# +
f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()


@jax.jit
def vf_2(y, dy, t):  # noqa: ARG001
    """Evaluate the three-body problem as a second-order IVP."""
    return f(y, dy, *f_args)


# One derivative more than above because we don't transform to first order
tcoeffs = taylor.odejet_padded_scan(lambda *ys: vf_2(*ys, t=t0), (u0, du0), num=3)
init, ibm, ssm = probdiffeq.prior_wiener_integrated(
    tcoeffs, output_scale=1.0, ssm_fact="isotropic"
)
ts0 = probdiffeq.correction_ts0(vf_2, ode_order=2, ssm=ssm)
strategy = probdiffeq.strategy_filter(ssm=ssm)
solver_2nd = probdiffeq.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
errorest_2nd = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)

# -

solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, errorest=errorest_2nd)
solution = jax.jit(solve)(init, save_at=save_at, atol=1e-5, rtol=1e-5)

norm = jnp.linalg.norm((solution.u.mean[0][-1, ...] - u0) / jnp.abs(1.0 + u0))
plt.title(f"error={norm:.3f}")
plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
plt.show()

# The results are indistinguishable from the plot.
# While the runtimes of both solvers are similar,
# the error of the second-order solver is much lower.
#
# See the benchmarks for more quantitative versions of this statement.
