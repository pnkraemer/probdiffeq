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

# Taylor-series: Neural ODE problem

```python
"""Benchmark all Taylor-series estimators on a Neural ODE."""

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
        return {"color": "C0", "linestyle": "solid"}
    if "forward" in label.lower():
        return {"color": "C1", "linestyle": "dashed", "label": label}
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis_compile, axis_perform, results):
    """Plot the results."""
    style_curve = {"alpha": 0.85}
    style_area = {"alpha": 0.15}
    for label, wp in results.items():
        style = choose_style(label)

        inputs = wp["arguments"]
        work_compile = wp["work_compile"]
        work_mean, work_std = wp["work_mean"], wp["work_std"]

        if "doubling" in label:
            num_repeats = jnp.diff(jnp.concatenate((jnp.ones((1,)), inputs)))
            inputs = jnp.arange(1, jnp.amax(inputs) * 1)
            work_compile = _adaptive_repeat(work_compile, num_repeats)
            work_mean = _adaptive_repeat(work_mean, num_repeats)
            work_std = _adaptive_repeat(work_std, num_repeats)
            # axis_perform.set_xticks(inputs[::2])

        axis_compile.semilogy(inputs, work_compile, **style, **style_curve)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis_perform.semilogy(inputs, work_mean, **style, **style_curve)
        axis_perform.fill_between(
            inputs, range_lower, range_upper, **style, **style_area
        )

    axis_compile.set_xticks(range(1, 15))
    axis_compile.set_ylim((1e-3, 1e2))
    return axis_compile, axis_perform


def _adaptive_repeat(xs, ys):
    """Repeat the doubling values correctly to create a comprehensible plot."""
    zs = []
    for x, y in zip(xs, ys):
        zs.extend([x] * int(y))
    return jnp.asarray(zs)
```

```python
plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())
```

```python
fig, (axis_perform, axis_compile) = plt.subplots(
    ncols=2, figsize=(8, 3), dpi=150, sharex=True
)

results = load_results()
axis_compile, axis_perform = plot_results(axis_compile, axis_perform, results)

axis_compile.set_title("Compilation time")
axis_perform.set_title("Evaluation time")
axis_compile.legend(loc="lower right")
axis_compile.set_xlabel("Number of Derivatives")
axis_perform.set_xlabel("Number of Derivatives")
axis_perform.set_ylabel("Wall time (sec)")
axis_perform.grid()
axis_compile.grid()


plt.show()
```

```python

```
