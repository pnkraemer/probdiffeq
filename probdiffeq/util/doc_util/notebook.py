"""Benchmark utils."""

import numpy as np
from tueplots import axes, cycler, fontsizes, markers


def plot_style():
    colors = ["cornflowerblue", "salmon", "mediumseagreen", "crimson", "darkorchid"]
    markers_ = ["o", "v", "P", "^", "X", "d"]
    return {
        **axes.color(base="black"),
        **axes.lines(base_width=0.5),
        **axes.tick_direction(x="inout", y="inout"),
        **axes.legend(),
        **axes.grid(grid_linestyle="dotted"),
        **cycler.cycler(
            marker=np.tile(markers_, 9)[:15], color=np.tile(colors, 10)[:15]
        ),
        **markers.with_edge(),
        **{"figure.dpi": 100},
    }


def plot_sizes():
    return fontsizes.beamer()
