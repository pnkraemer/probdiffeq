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

# Taylor-series: Pleiades

The Pleiades problem is a common non-stiff differential equation.

```python
"""Benchmark all Taylor-series estimators on the Pleiades problem."""

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
    if "doubling" in label.lower():
        return {"color": "C3", "linestyle": "dotted", "label": label}
    if "unroll" in label.lower():
        return {"color": "C2", "linestyle": "dashdot", "label": label}
    if "taylor" in label.lower():
        return {"color": "C0", "linestyle": "solid", "label": label}
    if "forward" in label.lower():
        return {"color": "C1", "linestyle": "dashed", "label": label}
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis_compile, axis_perform, results):
    """Plot the results."""
    for label, wp in results.items():
        style = choose_style(label)

        inputs = wp["arguments"]
        work_mean = wp["work_compile"]
        axis_compile.semilogy(inputs, work_mean, **style)

        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis_perform.semilogy(inputs, work_mean, **style)
        axis_perform.fill_between(inputs, range_lower, range_upper, alpha=0.3, **style)

    return axis_compile, axis_perform
```

```python
plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())
```

```python
fig, (axis_perform, axis_compile) = plt.subplots(
    ncols=2, dpi=150, sharex=True, figsize=(8, 3)
)

results = load_results()

axis_compile, axis_perform = plot_results(axis_compile, axis_perform, results)

axis_compile.set_title("Compilation time")
axis_perform.set_title("Evaluation time")
axis_compile.legend()
axis_compile.set_xlabel("Number of Derivatives")
axis_perform.set_xlabel("Number of Derivatives")
axis_perform.set_ylabel("Wall time (sec)")
axis_perform.grid()
axis_compile.grid()
axis_perform.set_yticks((1e-5, 1e-4))


plt.show()
```

```python

```
