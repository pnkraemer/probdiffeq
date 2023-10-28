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

# Hires

The HIRES problem is a common stiff differential equation.

```python
"""Benchmark all solvers on the HIRES problem."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config

from probdiffeq.util.doc_util import notebook

config.update("jax_platform_name", "cpu")
```

```python
def load_results():
    """Load the results from a file."""
    return jnp.load("./results.npy", allow_pickle=True)[()]


def load_solution():
    """Load the solution-to-be-plotted from a file."""
    ts = jnp.load("./plot_ts.npy")
    ys = jnp.load("./plot_ys.npy")
    return ts, ys


def choose_style(label):
    """Choose a plotting style for a given algorithm."""
    if "TS" in label:
        return {"color": "C0", "linestyle": "solid"}
    if "SciPy" in label:
        return {"color": "C2", "linestyle": "dashed"}
    if "iffrax" in label:
        return {"color": "C3", "linestyle": "dotted"}
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("Benchmark")
    for label, wp in results.items():
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_xlabel("Precision [relative RMSE]")
    axis.set_ylabel("Work [wall time, s]")
    axis.grid()
    axis.set_ylim((1e-4, 4e0))

    axis.legend(loc="upper center", ncols=3, mode="expand", facecolor="ghostwhite")
    return axis


def plot_solution(axis, ts, ys, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("Hires")
    kwargs = {"color": "black", "alpha": 0.85}

    axis.plot(ts, ys, linestyle="dashed", marker="None", **kwargs)
    for y in ys.T:
        axis.plot(ts[0], y[0], linestyle="None", marker=".", markersize=4, **kwargs)
        axis.plot(ts[-1], y[-1], linestyle="None", marker=".", markersize=4, **kwargs)

    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis
```

```python
plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())
```

```python
layout = [
    ["benchmark", "benchmark", "solution"],
    ["benchmark", "benchmark", "solution"],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 3), constrained_layout=True, dpi=300)


results = load_results()
ts, ys = load_solution()


_ = plot_results(axes["benchmark"], results)
_ = plot_solution(axes["solution"], ts, ys)
plt.show()
```
