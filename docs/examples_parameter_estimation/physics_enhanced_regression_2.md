---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Parameter estimation with BlackJAX


This tutorial explains how to estimate unknown parameters of initial value problems (IVPs) using Markov Chain Monte Carlo (MCMC) methods as provided by [BlackJAX](https://blackjax-devs.github.io/blackjax/).

## TL;DR
Compute log-posterior of IVP parameters given observations of the IVP solution with ProbDiffEq. Sample from this posterior using BlackJAX.
Evaluating the log-likelihood of the data is described [in this paper](https://arxiv.org/abs/2202.01287). Based on this log-likelihood, sampling from the log-posterior is as done [in this paper](https://arxiv.org/abs/2002.09301).


## Technical setup
Let $f$ be a known vector field. In this example, we use the Lotka-Volterra model.
Consider an ordinary differential equation 

$$
\dot y(t) = f(y(t)), \quad 0 \leq t \leq T
$$

subject to an unknown initial condition $y(0) = \theta$.
Recall from the previous tutorials that the probabilistic IVP solution is an approximation of the posterior distribution

$$
p\left(y ~|~ [\dot y(t_n) = f(y(t_n))]_{n=0}^N, y(0) = \theta\right)
$$

for a Gaussian prior over $y$ and a pre-determined or adaptively selected grid $t_0, ..., t_N$.

We don't know the initial condition of the IVP, but assume that we have noisy observations of the IVP soution $y$ at the terminal time $T$ of the integration problem,

$$
p(\text{data}~|~ y(T)) = N(y(T), \sigma^2 I)
$$

for some $\sigma > 0$.
We can use these observations to reconstruct $\theta$, for example by sampling from $p(\text{data}~|~\theta)$ (which is a function of $\theta$).

Now, one way of evaluating $p(\text{data} ~|~ \theta)$ is to use any numerical solver, for example a Runge-Kutta method, to approximate $y(T)$ from $\theta$ and evaluate $N(y(T), \sigma^2 I)$.
But this ignores a few crucial concepts (e.g., the numerical error of the approximation; we refer to the references linked above). 
Instead, we can use a probabilistic solver instead of "any" numerical solver and build a more comprehensive model:

**We can combine probabilistic IVP solvers with MCMC methods to estimate $\theta$ from $\text{data}$ in a way that quantifies numerical approximation errors (and other model mismatches).**
To do so, we approximate the distribution of the IVP solution given the parameter $p(y(T) \mid \theta)$ and evaluate the marginal distribution of $N(y(T), \sigma^2I)$ given the probabilistic IVP solution.
More formally, we use ProbDiffEq to evaluate the density of the unnormalised posterior

$$
M(\theta) := p(\theta \mid \text{data})\propto p(\text{data} \mid \theta)p(\theta)
$$

where "$\propto$" means "proportional to" and the likelihood stems from the IVP solution

$$
p(\text{data} \mid \theta) = \int p(\text{data} \mid y(T)) p(y(T) \mid [\dot y(t_n) = f(y(t_n))]_{n=0}^N, y(0) = \theta), \theta) d y(T)
$$
Loosely speaking, this distribution averages $N(y(T), \sigma^2I)$ over all IVP solutions $y(T)$ that are realistic given the differential equation, grid $t_0, ..., t_N$, and prior distribution $p(y)$.
This is useful, because if the approximation error is large, $M(\theta)$ "knows this". If the approximation error is ridiculously small, $M(\theta)$ "knows this" too and  we recover the procedure described for non-probabilistic solvers above.
**Interestingly, non-probabilistic solvers cannot do this** averaging because they do not yield a statistical description of estimated IVP solutions.
Non-probabilistic solvers would also fail if the observations were noise-free (i.e. $\sigma = 0$), but the present example notebook remains stable. (Try it yourself!)



To sample $\theta$ according to $M$ (respectively $\log M$), we evaluate $M(\theta)$ with ProbDiffEq, compute its gradient with JAX, and use this gradient to sample $\theta$ with BlackJAX:

1. ProbDiffEq: Compute the probabilistic IVP solution by approximating $p(y(T) ~|~ [\dot y(t_n) = f(y(t_n))]_n, y(0) = \theta)$

2. ProbDiffEq: Evaluate  $M(\theta)$ by marginalising over the IVP solution computed in step 1.

3. JAX: Compute the gradient $\nabla_\theta M(\theta)$

4. BlackJAX: Sample from $\log M(\theta)$ using, for example, the No-U-Turn-Sampler (which requires $\nabla_\theta M(\theta))$.

Here is how:
<!-- #endregion -->

```python
"""Estimate ODE paramaters with ProbDiffEq and BlackJAX."""

import functools

import blackjax
import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import adaptive, ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from probdiffeq.util.doc_util import notebook
```

```python
# x64 precision
config.update("jax_enable_x64", True)

# CPU
config.update("jax_platform_name", "cpu")

# IVP examples in JAX
if not backend.has_been_selected:
    backend.select("jax")

# Nice-looking plots
plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())
```

```python

```

```python
impl.select("isotropic", ode_shape=(2,))
```

## Problem setting

First, we set up an IVP and create some artificial data by simulating the system with "incorrect" initial conditions.

```python
f, u0, (t0, t1), f_args = ivps.lotka_volterra()


@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the Lotka-Volterra vector field."""
    return f(y, *f_args)


theta_true = u0 + 0.5 * jnp.flip(u0)
theta_guess = u0  # initial guess
```

```python
def plot_solution(sol, *, ax, marker=".", **plotting_kwargs):
    """Plot the IVP solution."""
    for d in [0, 1]:
        ax.plot(sol.t, sol.u[:, d], marker="None", **plotting_kwargs)
        ax.plot(sol.t[0], sol.u[0, d], marker=marker, **plotting_kwargs)
        ax.plot(sol.t[-1], sol.u[-1, d], marker=marker, **plotting_kwargs)
    return ax


@jax.jit
def solve_fixed(theta, *, ts):
    """Evaluate the parameter-to-solution map, solving on a fixed grid."""
    # Create a probabilistic solver
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (theta,), num=2)
    output_scale = 10.0
    init = solver.initial_condition(tcoeffs, output_scale)

    sol = ivpsolve.solve_fixed_grid(vf, init, grid=ts, solver=solver)
    return sol[-1]


@jax.jit
def solve_adaptive(theta, *, save_at):
    """Evaluate the parameter-to-solution map, solving on an adaptive grid."""
    # Create a probabilistic solver
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver)

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (theta,), num=2)
    output_scale = 10.0
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.solve_and_save_at(
        vf, init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1
    )


save_at = jnp.linspace(t0, t1, num=250, endpoint=True)
solve_save_at = functools.partial(solve_adaptive, save_at=save_at)
```

```python
# Visualise the initial guess and the data

fig, ax = plt.subplots(figsize=(5, 3))

data_kwargs = {"alpha": 0.5, "color": "gray"}
ax.annotate("Data", (13.0, 30.0), **data_kwargs)
sol = solve_save_at(theta_true)
ax = plot_solution(sol, ax=ax, **data_kwargs)

guess_kwargs = {"color": "C3"}
ax.annotate("Initial guess", (7.5, 20.0), **guess_kwargs)
sol = solve_save_at(theta_guess)
ax = plot_solution(sol, ax=ax, **guess_kwargs)
plt.show()
```

## Log-posterior densities via ProbDiffEq

Set up a log-posterior density function that we can plug into BlackJAX. Choose a Gaussian prior centered at the initial guess with a large variance.

```python
mean = theta_guess
cov = jnp.eye(2) * 30  # fairly uninformed prior


@jax.jit
def logposterior_fn(theta, *, data, ts, obs_stdev=0.1):
    """Evaluate the logposterior-function of the data."""
    y_T = solve_fixed(theta, ts=ts)
    logpdf_data = solution.log_marginal_likelihood_terminal_values(
        data, standard_deviation=obs_stdev, posterior=y_T.posterior
    )
    logpdf_prior = jax.scipy.stats.multivariate_normal.logpdf(theta, mean=mean, cov=cov)
    return logpdf_data + logpdf_prior


# Fixed steps for reverse-mode differentiability:


ts = jnp.linspace(t0, t1, endpoint=True, num=100)
data = solve_fixed(theta_true, ts=ts).u

log_M = functools.partial(logposterior_fn, data=data, ts=ts)
```

```python
print(jnp.exp(log_M(theta_true)), ">=", jnp.exp(log_M(theta_guess)), "?")
```

## Sampling with BlackJAX

From here on, BlackJAX takes over:


Set up a sampler.

```python
@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run BlackJAX' inference loop."""

    def one_step(state, rng_key):
        state, _ = kernel.step(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

Initialise the sampler, warm it up, and run the inference loop.

```python
initial_position = theta_guess
rng_key = jax.random.PRNGKey(0)
```

```python
# WARMUP
warmup = blackjax.window_adaptation(blackjax.nuts, log_M, progress_bar=True)

warmup_results, _ = warmup.run(rng_key, initial_position, num_steps=200)

initial_state = warmup_results.state
step_size = warmup_results.parameters["step_size"]
inverse_mass_matrix = warmup_results.parameters["inverse_mass_matrix"]
nuts_kernel = blackjax.nuts(
    logdensity_fn=log_M, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
)
```

```python
# INFERENCE LOOP
rng_key, _ = jax.random.split(rng_key, 2)
states = inference_loop(
    rng_key, kernel=nuts_kernel, initial_state=initial_state, num_samples=150
)
```

## Visualisation

Now that we have samples of $\theta$, let's plot the corresponding solutions:

```python
solution_samples = jax.vmap(solve_save_at)(states.position)
```

```python
# Visualise the initial guess and the data

fig, ax = plt.subplots()

sample_kwargs = {"color": "C0"}
ax.annotate("Samples", (2.75, 31.0), **sample_kwargs)
for sol in solution_samples:
    ax = plot_solution(sol, ax=ax, linewidth=0.1, alpha=0.75, **sample_kwargs)

data_kwargs = {"color": "gray"}
ax.annotate("Data", (18.25, 40.0), **data_kwargs)
sol = solve_save_at(theta_true)
ax = plot_solution(sol, ax=ax, linewidth=4, alpha=0.5, **data_kwargs)

guess_kwargs = {"color": "gray"}
ax.annotate("Initial guess", (6.0, 12.0), **guess_kwargs)
sol = solve_save_at(theta_guess)
ax = plot_solution(sol, ax=ax, linestyle="dashed", alpha=0.75, **guess_kwargs)
plt.show()
```

The samples cover a perhaps surpringly large range of potential initial conditions, but lead to the "correct" data.

In parameter space, this is what it looks like:

```python
plt.title("Posterior samples (parameter space)")
plt.plot(states.position[:, 0], states.position[:, 1], "o", alpha=0.5, markersize=4)
plt.plot(theta_true[0], theta_true[1], "P", label="Truth", markersize=8)
plt.plot(theta_guess[0], theta_guess[1], "P", label="Initial guess", markersize=8)
plt.legend()
plt.show()
```

Let's add the value of $M$ to the plot to see whether the sampler covers the entire region of interest.

```python
xlim = 17, jnp.amax(states.position[:, 0]) + 0.5
ylim = 17, jnp.amax(states.position[:, 1]) + 0.5

xs = jnp.linspace(*xlim, endpoint=True, num=300)
ys = jnp.linspace(*ylim, endpoint=True, num=300)
Xs, Ys = jnp.meshgrid(xs, ys)

Thetas = jnp.stack((Xs, Ys))
log_M_vmapped_x = jax.vmap(log_M, in_axes=-1, out_axes=-1)
log_M_vmapped = jax.vmap(log_M_vmapped_x, in_axes=-1, out_axes=-1)
Zs = log_M_vmapped(Thetas)
```

```python
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3))

ax_samples, ax_heatmap = ax

fig.suptitle("Posterior samples (parameter space)")
ax_samples.plot(
    states.position[:, 0], states.position[:, 1], ".", alpha=0.5, markersize=4
)
ax_samples.plot(theta_true[0], theta_true[1], "P", label="Truth", markersize=8)
ax_samples.plot(
    theta_guess[0], theta_guess[1], "P", label="Initial guess", markersize=8
)
ax_samples.legend()
im = ax_heatmap.contourf(Xs, Ys, jnp.exp(Zs), cmap="cividis", alpha=0.8)
plt.colorbar(im)
plt.show()
```

<!-- #region -->
Looks great!

## Conclusion

In conclusion, a log-posterior density function can be provided by ProbDiffEq such that any of BlackJAX' samplers yield parameter estimates of IVPs. 


## What's next

Try to get a feeling for how the sampler reacts to changing observation noises, solver parameters, and so on.
We could extend the sampling problem from $\theta \mapsto \log M(\theta)$ to some $(\theta, \sigma) \mapsto \log \tilde M(\theta, \sigma)$, i.e., treat the observation noise as unknown and run Hamiltonian Monte Carlo in a higher-dimensional parameter space.
We could also add a more suitable prior distribution $p(\theta)$ to regularise the problem.


A final side note:
We could also replace the sampler with an optimisation algorithm and use this procedure to solve boundary value problems (albeit this may not be very efficient; use [this](https://arxiv.org/abs/2106.07761) algorithm instead).
<!-- #endregion -->
