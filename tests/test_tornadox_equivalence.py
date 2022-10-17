"""Make sure that the results are equivalent to tornadox."""

import jax
import jax.numpy as jnp
from diffeqzoo import ivps
from tornadox import ek0, init, ivp, step

from odefilter import controls, ivpsolve, recipes

# todo: write the EKF1 test. It is also possible already!


def test_kronecker_ek0():

    # van-der-Pol as a setup. We really don't want stiffness here.
    f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order(
        time_span=(0.0, 1.0), stiffness_constant=1.0
    )

    @jax.jit
    def vf(t, y):
        return f(y, *f_args)

    num = 4  # this is the default argument
    # stricter than default, because we want to reject steps!
    atol, rtol = (1e-4, 1e-6)
    factor_min, factor_max, safety = 0.2, 10.0, 0.95  # default

    # Tornadox. These arguments are defaults, but we fix them anyway.
    solver = ek0.KroneckerEK0(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=step.AdaptiveSteps(
            max_changes=(factor_min, factor_max),
            safety_scale=safety,
            abstol=atol,
            reltol=rtol,
        ),
    )
    vdp = ivp.InitialValueProblem(f=vf, t0=t0, tmax=t1, y0=u0)
    solution_tornadox = solver.solve(vdp)

    # ODE-filter. We need to use a very specific setup.
    ekf0, info_op = recipes.dynamic_isotropic_ekf0(num_derivatives=num)
    controller = controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )
    solution_odefilter = ivpsolve.solve(
        vf,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        solver=ekf0,
        info_op=info_op,
        atol=atol,
        rtol=rtol,
        control=controller,
    )

    # Compare t, mean, cov_sqrtm @ cov_sqrtm.T.
    # They should be _identical_ (up to machine precision).

    @jax.vmap
    def cov(x):
        return x @ x.T

    d = u0.shape[0]

    @jax.vmap
    def kroncov(x):
        return jnp.kron(jnp.eye(d), x @ x.T)

    assert jnp.allclose(solution_tornadox.t, solution_odefilter.t)
    assert jnp.allclose(solution_tornadox.mean, solution_odefilter.marginals.mean)
    assert jnp.allclose(
        cov(solution_tornadox.cov_sqrtm),
        kroncov(solution_odefilter.marginals.cov_sqrtm_lower),
    )
