"""Benchmark utils."""

import numpy as np
from tueplots import axes, cycler, markers


def plot_config():
    colors = ["cornflowerblue", "salmon", "mediumseagreen", "crimson", "darkorchid"]
    markers_ = ["o", "v", "P", "^", "X", "d"]
    return {
        **axes.color(base="black"),
        **axes.lines(base_width=0.5),
        **axes.legend(),
        **axes.grid(),
        **cycler.cycler(
            marker=np.tile(markers_, 9)[:15],
            color=np.tile(colors, 10)[:15],
        ),
        **markers.with_edge(),
        **{"figure.dpi": 100},
    }
