"""Benchmark utils."""

import numpy as np
from tueplots import axes, cycler, figsizes, fontsizes, markers


def plot_config():
    colors = ["cornflowerblue", "salmon", "mediumseagreen", "crimson", "darkorchid"]
    markers_ = ["o", "v", "P", "^", "X", "d"]
    return {
        **figsizes.beamer_169(),
        **fontsizes.beamer_moml(),
        **axes.color(base="black"),
        **axes.lines(),
        **axes.legend(),
        **axes.grid(),
        **cycler.cycler(
            marker=np.tile(markers_, 9)[:15],
            color=np.tile(colors, 10)[:15],
        ),
        **markers.with_edge(),
        **{"figure.dpi": 120},
    }
