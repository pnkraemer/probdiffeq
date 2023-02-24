"""Benchmark utils."""

import numpy as np
from tueplots import axes, bundles, cycler, markers


def plot_config():
    colors = ["cornflowerblue", "salmon", "mediumseagreen", "crimson", "darkorchid"]
    markers_ = ["o", "v", "P", "^", "X", "d"]
    return {
        **bundles.beamer_moml(rel_width=1.0, rel_height=1.0),
        **axes.color(base="black"),
        **axes.grid(),
        **cycler.cycler(
            marker=np.tile(markers_, 9)[:15],
            color=np.tile(colors, 10)[:15],
        ),
        **markers.with_edge(),
        "figure.dpi": 150,
    }
