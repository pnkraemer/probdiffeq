from probdiffeq.backend import functools, tree_array_util, warnings
from probdiffeq.ivpsolve import (
    RejectionLoop,
    Solution,
    T,
    control_proportional_integral,
)


def solve_adaptive_save_every_step(solver, errorest, control=None, clip_dt=False):
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
        msg = f"Strategy {solver} should not be used in solve_adaptive_save_every_step."
        warnings.warn(msg, stacklevel=1)
    if control is None:
        control = control_proportional_integral()

    loop = RejectionLoop(
        solver=solver, clip_dt=clip_dt, control=control, errorest=errorest
    )

    def solve(u: T, t0, t1, *, atol, rtol, dt0=0.1, eps=1e-8) -> Solution[T]:
        solution0 = functools.jit(solver.init)(t=t0, u=u)
        state = functools.jit(loop.init)(solution0, dt=dt0)

        rejection_loop_apply = functools.jit(loop.loop)

        solutions = []
        while state.step_from.t < t1:
            solution, state = rejection_loop_apply(
                state, t1=t1, eps=eps, atol=atol, rtol=rtol
            )
            solutions.append(solution)

        solutions = tree_array_util.tree_stack(solutions)
        return functools.jit(solver.userfriendly_output)(
            solution0=solution0, solution=solutions
        )

    return solve
