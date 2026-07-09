"""Test that dynamic calibration is consistent across factorisations.

E.g. for isotropic VFs, dense and isotropic solvers should be *identical*.
"""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing


@testing.parametrize("num_derivatives", [2])
@testing.parametrize("ode_shape", [(5,)])
def test_dense_vs_isotropic(num_derivatives, ode_shape):
    """Assert that for an isotropic ODE, dense solvers = isotropic solvers."""
    # Isotropic ODE

    key = random.prng_key(seed=1)
    key, subkey = random.split(key, num=2)
    scalar = random.normal(subkey, shape=())

    key, subkey = random.split(key, num=2)
    u0 = random.normal(subkey, shape=ode_shape)

    t0 = 0.0
    t1 = 10.0

    @func.partial(probdiffeq.ode, jacobian=probdiffeq.jacobian_materialize())
    def vf(u, *, t):
        return scalar * u + t

    # Generate a solver (common elements)
    ts = np.linspace(t0, t1, num=3, endpoint=True)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)

    # Generate two solutions: dense and isotropic
    sols = []
    scales = []
    for ssm in [
        probdiffeq.state_space_model_dense(),
        probdiffeq.state_space_model_isotropic(),
    ]:
        iwp = ssm.prior_wiener_integrated(tcoeffs)

        # Compute a reference solution
        constraint = ssm.constraint_ode_ts1(vf.jet_lift_max(num_tcoeffs=len(tcoeffs)))
        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=constraint)
        error = probdiffeq.error_state_std(constraint=constraint)
        solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

        # Solutions should satisfy "allclose" despite this looooose tolerance:
        solution = func.jit(solve)(iwp, save_at=ts, atol=1e-1, rtol=1e-1)
        sols.append(solution.u.to_multivariate_normal())
        scales.append(solution.output_scale)

    sc1, sc2 = scales
    assert testing.allclose(sc1, sc2)

    (m1, C1), (m2, C2) = sols
    assert testing.allclose(m1, m2)

    assert testing.allclose(C1, C2)


@testing.parametrize("ode_shape", [(5,)])
def test_dense_vs_blockdiag(ode_shape):
    """Assert that solving (d,) scalar ODEs in parallel with dense solvers matches a single blockdiagonal solve.

    For instance, this catches issues like normalising the RMS computation in the blockdiagonal
    model with the size of the wrong array (which happened in the past).

    We do not test adaptive steps because since the blockdiagonal solver
    carries blockdiagonal output scales, the step-size mechanics are generally
    different between blockdiagonal and dense solvers, even if the ODE has a diagonal Jacobian.
    """
    key = random.prng_key(seed=1)
    key, subkey = random.split(key, num=2)
    diagonal = random.normal(subkey, shape=ode_shape)

    key, subkey = random.split(key, num=2)
    u0 = random.normal(subkey, shape=ode_shape)

    t0, t1 = 0.0, 10.0
    ts = np.linspace(t0, t1, num=10, endpoint=True)

    solve_dense, solve_blockdiag = _make_solvers(ts=ts)
    sol_dense = func.vmap(solve_dense)(diagonal, u0)
    sol_blockdiag = solve_blockdiag(diagonal, u0)

    # Swap the axes for the batched solver: turn (D, T, N) into (T, D, N)
    # in order to be able to compare to the blockdiagonal solver
    mean_dense = linalg.einsum("ij...->ji...", sol_dense.u.mean_flat)
    cholesky_dense = linalg.einsum("ij...->ji...", sol_dense.u.cholesky_flat)
    mean_blockdiag = sol_blockdiag.u.mean_flat
    cholesky_blockdiag = sol_blockdiag.u.cholesky_flat
    assert testing.allclose(mean_dense, mean_blockdiag)
    assert testing.allclose(cholesky_dense, cholesky_blockdiag)


def _make_solvers(*, ts):
    """Create two solvers: (i) a block-diagonal solver, and (ii) a dense solver.

    With fixed steps, vmap(dense)() should be identical to blockdiag()
    as soon as the vector field treats each component independently.
    """
    strategy = probdiffeq.strategy_smoother_fixedinterval()
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=3)

    @func.jit
    def solve_dense(scalar, u0):
        """Solve a scalar ODE with a dense SSM."""

        @func.partial(probdiffeq.ode, jacobian=probdiffeq.jacobian_materialize())
        def vf(u, *, t):
            return scalar * u + t

        tcoeffs, _ = jetexpand(vf, [u0], t=ts[0])

        ssm = probdiffeq.state_space_model_dense()
        iwp = ssm.prior_wiener_integrated(tcoeffs)
        constraint = ssm.constraint_ode_ts1(vf.jet_lift_max(num_tcoeffs=len(tcoeffs)))

        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=constraint)
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        return solve(iwp, grid=ts)

    @func.jit
    def solve_blockdiag(diagonal, u0):
        """Solve a diagonal ODE with a (block)diagonal SSM."""

        @func.partial(probdiffeq.ode, jacobian=probdiffeq.jacobian_materialize())
        def vf(u, *, t):
            return diagonal * u + t

        tcoeffs, _ = jetexpand(vf, [u0], t=ts[0])

        ssm = probdiffeq.state_space_model_blockdiag()
        iwp = ssm.prior_wiener_integrated(tcoeffs)
        constraint = ssm.constraint_ode_ts1(vf.jet_lift_max(num_tcoeffs=len(tcoeffs)))

        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=constraint)
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        return solve(iwp, grid=ts)

    return solve_dense, solve_blockdiag
