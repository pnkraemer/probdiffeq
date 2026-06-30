from probdiffeq._ivpsolve import solver_protocols
from probdiffeq.backend import flow, np, warnings
from probdiffeq.backend.typing import Callable, TypeVar

T = TypeVar("T")
S = TypeVar("S")

__all__ = ["solve_fixed_grid"]


def solve_fixed_grid(
    *, solver: solver_protocols.Solver
) -> Callable[..., solver_protocols.Solution]:
    """Solve an initial value problem on a fixed, pre-determined grid."""
    if not solver.is_suitable_for_save_every_step:
        msg = f"Solver {solver} should not be used in solve_adaptive_save_every_step/solve_fixed_grid."
        msg += " This is typically caused by using a fixed-point smoother."
        msg += " Try using filters or fixed-interval smoothers instead."
        warnings.warn(msg, stacklevel=1)

    def solve(u: T, /, *, grid, damp: float = 0.0) -> solver_protocols.Solution[T]:
        def body_fn(s, dt):
            s_new = solver.step(state=s, dt=dt, damp=damp)
            return s_new, s_new

        t0 = grid[0]
        state0 = solver.init(t=t0, u=u, damp=damp)
        s_new, result = flow.scan(body_fn, init=state0, xs=np.diff(grid))

        return solver.userfriendly_output(
            solution0=state0, solution=result, solution1=s_new
        )

    return solve
