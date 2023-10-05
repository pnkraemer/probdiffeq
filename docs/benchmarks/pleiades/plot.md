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

# Pleiades

The Pleiades problem is a common non-stiff differential equation.

```python
"""Benchmark all solvers on the Pleiades problem."""

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


def choose_style(label):
    """Choose a plotting style for a given algorithm."""
    if "probdiffeq" in label.lower():
        return {"color": "C0", "linestyle": "solid"}
    if "numba" in label.lower():
        return {"color": "C4", "linestyle": "dashed"}
    if "scipy" in label.lower():
        return {"color": "C2", "linestyle": "dashed"}
    if "diffrax" in label.lower():
        return {"color": "C3", "linestyle": "dotted"}
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis, results):
    """Plot the results."""
    for label, wp in results.items():
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_xlabel("Precision [absolute RMSE]")
    axis.set_ylabel("Work [wall time, s]")
    axis.grid()
    return axis
```

```python
plt.rcParams.update(notebook.plot_config())

fig, axis = plt.subplots(dpi=150)
fig.suptitle("Pleiades problem, terminal-value simulation")

results = load_results()
axis = plot_results(axis, results)
axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
```

```python

```
