"""BVP solver."""

from probdiffeq.statespace.scalar import extra


def solve_fixed_grid(vf, bcond, grid):
    extra = _extrapolation_model(grid)

    corr_bcond = _correction_model_bcond(bcond, grid)
    data_bcond = _zeros_padded(grid)
    extra_bcond = _kalmanfilter(extra, corr_bcond, data_bcond, reverse=True)

    corr_ode = _correction_model_ode(vf, grid)
    data_ode = _zeros(grid)
    solution = _kalmanfilter(extra_bcond, corr_ode, data_ode, reverse=False)

    return solution


def _extrapolation_model(grid):
    pass


def _correction_model_bcond(bcond, grid):
    pass


def _zeros_padded(grid):
    pass


def _correction_model_ode(vf, mesh):
    pass


def _zeros(grid):
    pass


def _kalmanfilter(extra, corr, data, reverse=False):
    pass
