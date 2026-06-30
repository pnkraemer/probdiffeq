"""Test utilities."""

import probdiffeq.ivpsolve
from probdiffeq.backend import flow, func, linalg, np, tree, warnings
from probdiffeq.backend.typing import Callable, TypeVar

T = TypeVar("T")


def solve_adaptive_save_every_step(solver, error, control=None, clip_dt=False):
    """Solve an initial value problem and save every step.

    This function uses a native-Python while loop.

    !!! warning
        Not JITable, not reverse-mode-differentiable.

    Since it's not really compatible with JAX's function transformations,
    solve-adaptive-save-every-step is not a part of the IVP solver suite.
    However, since it's really useful for unittests, we offer an implementation in this
    test-utilities module.
    """
    if not solver.is_suitable_for_save_every_step:
        msg = f"Solver {solver} should not be used in solve_adaptive_save_every_step."
        msg += " This is typically caused by the wrong strategy selection."
        msg += " Try using filters or fixed-interval smoothers."
        warnings.warn(msg, stacklevel=1)

    if control is None:
        control = probdiffeq.ivpsolve.control_integral()

    loop = probdiffeq.ivpsolve.RejectionLoop(
        solver=solver,
        clip_dt=clip_dt,
        control=control,
        error=error,
        # We do not expose this option to the user
        # because we do not want to suggest that this function
        # uses meaningful looping to begin with.
        while_loop=flow.while_loop,
    )

    def solve(
        u: T, t0, t1, *, atol, rtol, dt0=0.1, eps=1e-8, damp=0.0
    ) -> probdiffeq.ivpsolve.Solution[T]:
        t0 = np.asarray(t0)
        t1 = np.asarray(t1)

        solution0 = func.jit(solver.init)(t=t0, u=u, damp=damp)
        state = func.jit(loop.init)(solution0, dt=dt0)

        rejection_loop_apply = func.jit(loop.loop)

        solutions = []
        while state.step_from.t < t1:
            solution, state = rejection_loop_apply(
                state, t1=t1, eps=eps, atol=atol, rtol=rtol, damp=damp
            )
            solutions.append(solution)

        solutions = tree.tree_array_stack(solutions)
        return func.jit(solver.userfriendly_output)(
            solution0=solution0, solution=solutions, solution1=state.step_from
        )

    return solve


def exp_gram_matrix_fraction(expm=linalg.expm) -> Callable:
    """Compute matrix exponential and finite-horizon Gramian via matrix fractions."""

    def compute(A, B):
        n = A.shape[0]
        M = np.block([[A, B @ B.T], [np.zeros_like(A), -A.T]])
        E = expm(M)
        eA = E[:n, :n]
        Sigma = E[:n, n:] @ eA.T
        return eA, Sigma

    return compute
