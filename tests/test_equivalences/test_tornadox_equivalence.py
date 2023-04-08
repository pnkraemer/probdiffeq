"""Make sure that the results are equivalent to tornadox."""

import diffeqzoo.ivps
import jax
import jax.numpy as jnp
from tornadox import ek0, ek1, init, ivp, step

from probdiffeq import controls, ivpsolve, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters


@testing.fixture(name="num")
def fixture_num():
    return 4


@testing.fixture(name="control_params")
def fixture_control_params():
    return 0.2, 10.0, 0.95


@testing.fixture(name="vanderpol")
def fixture_vanderpol():
    # van-der-Pol as a setup. We really don't want stiffness here.
    return diffeqzoo.ivps.van_der_pol_first_order(
        time_span=(0.0, 1.0), stiffness_constant=1.0
    )


@testing.fixture(name="ivp_tornadox")
def fixture_ivp_tornadox(vanderpol):
    f, u0, (t0, t1), f_args = vanderpol

    @jax.jit
    def vf_tor(t, y):  # pylint: disable=unused-argument
        return f(y, *f_args)

    return ivp.InitialValueProblem(
        f=vf_tor, t0=t0, tmax=t1, y0=u0, df=jax.jit(jax.jacfwd(vf_tor, argnums=1))
    )


@testing.fixture(name="ivp_probdiffeq")
def fixture_ivp_probdiffeq(vanderpol):
    f, u0, (t0, t1), f_args = vanderpol

    # ODE-filter
    @jax.jit
    def vf_ode(y, *, t, p):  # pylint: disable=unused-argument
        return f(y, *p)

    return vf_ode, (u0,), (t0, t1), f_args


@testing.fixture(name="steprule_tornadox")
def fixture_steprule_tornadox(solver_config, control_params):
    factor_min, factor_max, safety = control_params
    return step.AdaptiveSteps(
        max_changes=(factor_min, factor_max),
        safety_scale=safety,
        abstol=solver_config.atol_solve,
        reltol=solver_config.rtol_solve,
    )


@testing.fixture(name="controller_probdiffeq")
def fixture_controller_probdiffeq(control_params):
    factor_min, factor_max, safety = control_params
    return controls.IntegralClipped(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )


@testing.fixture(name="solver_tornadox_kronecker_ek0")
def fixture_solver_tornadox_kronecker_ek0(num, steprule_tornadox):
    return ek0.KroneckerEK0(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=steprule_tornadox,
    )


@testing.fixture(name="solver_probdiffeq_kronecker_ek0")
def fixture_solver_probdiffeq_kronecker_ek0(num):
    implementation = recipes.ts0_iso(num_derivatives=num)
    strategy = filters.Filter(implementation=implementation)
    return ivpsolvers.DynamicSolver(strategy=strategy)


@testing.case
def case_solver_pair_kronecker_ek0(
    solver_config,
    ivp_tornadox,
    ivp_probdiffeq,
    solver_tornadox_kronecker_ek0,
    solver_probdiffeq_kronecker_ek0,
    controller_probdiffeq,
):
    # Solve with tornadox
    solution_tornadox = solver_tornadox_kronecker_ek0.solve(ivp_tornadox)

    # Solve with probdiffeq
    vf_ode, u0, (t0, t1), f_args = ivp_probdiffeq
    solution_probdiffeq = ivpsolve.solve_with_python_while_loop(
        vf_ode,
        initial_values=u0,
        t0=t0,
        t1=t1,
        solver=solver_probdiffeq_kronecker_ek0,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        control=controller_probdiffeq,
        parameters=f_args,
        propose_dt0_nugget=0.0,
        reference_state_fn=lambda x, y: jnp.abs(x),
    )

    # Get both solutions into the same format

    @jax.vmap
    def cov(x):
        return x @ x.T

    d = u0[0].shape[0]

    @jax.vmap
    def kroncov(x):
        return jnp.kron(jnp.eye(d), x @ x.T)

    output_tornadox = (
        solution_tornadox.t,
        solution_tornadox.mean,
        cov(solution_tornadox.cov_sqrtm),
    )
    solution_probdiffeq = (
        solution_probdiffeq.t,
        solution_probdiffeq.marginals.hidden_state.mean,
        kroncov(solution_probdiffeq.marginals.hidden_state.cov_sqrtm_lower),
    )
    return output_tornadox, solution_probdiffeq


@testing.fixture(name="solver_tornadox_reference_ek1")
def fixture_solver_tornadox_reference_ek1(num, steprule_tornadox):
    return ek1.ReferenceEK1(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=steprule_tornadox,
    )


@testing.fixture(name="solver_probdiffeq_reference_ek1")
def fixture_solver_probdiffeq_reference_ek1(num):
    implementation = recipes.ts1_dense(num_derivatives=num, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return ivpsolvers.DynamicSolver(strategy=strategy)


@testing.case
def case_solver_pair_reference_ek1(
    solver_config,
    ivp_tornadox,
    ivp_probdiffeq,
    solver_tornadox_reference_ek1,
    solver_probdiffeq_reference_ek1,
    controller_probdiffeq,
):
    # Solve with tornadox
    solution_tornadox = solver_tornadox_reference_ek1.solve(ivp_tornadox)

    # Solve with probdiffeq
    vf_ode, u0, (t0, t1), f_args = ivp_probdiffeq
    solution_probdiffeq = ivpsolve.solve_with_python_while_loop(
        vf_ode,
        initial_values=u0,
        t0=t0,
        t1=t1,
        solver=solver_probdiffeq_reference_ek1,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        control=controller_probdiffeq,
        parameters=f_args,
        initial_dt_nugget=0.0,
    )

    # Get both into the same format

    @jax.vmap
    def cov(x):
        return x @ x.T

    @jax.vmap
    def kronmean(x):
        return jnp.reshape(x, (-1,), order="F")

    output_tornadox = (
        solution_tornadox.t,
        kronmean(solution_tornadox.mean),
        cov(solution_tornadox.cov_sqrtm),
    )
    solution_probdiffeq = (
        solution_probdiffeq.t,
        solution_probdiffeq.marginals.hidden_state.mean,
        cov(solution_probdiffeq.marginals.hidden_state.cov_sqrtm_lower),
    )
    return output_tornadox, solution_probdiffeq


@testing.parametrize_with_cases(
    "solution_tornadox, solution_probdiffeq", cases=".", prefix="case_solver_pair_"
)
def test_outputs_equal(solution_tornadox, solution_probdiffeq):
    for sol_tornadox, sol_probdiffeq in zip(solution_tornadox, solution_probdiffeq):
        # We need to lower the 'rtol' because tornadox is incorrect:
        # To estimate errors, both probdiffeq and tornadox compute whitened residuals
        # as (L^T)^(-1)v. But tornadox calls solve_triangular incorrectly
        # (it forgets to set trans=1.)
        # ProbDiffEq does it correctly as confirmed by the logpdf-tests
        # (which fail with a small but noticeable error without  trans=1)
        #
        # Honestly, I find it a bit strange that the effect of this incorrect call
        #  is not bigger. The output scales are simply wrong.
        #  But until we run into more issues, let's roll with it.
        assert jnp.allclose(sol_tornadox, sol_probdiffeq, rtol=2e-2)
